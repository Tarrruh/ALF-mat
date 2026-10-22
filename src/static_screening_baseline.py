"""
Static (non-sequential) screening baseline, for comparison with sequential learning (SL).

Same setup as the SL study, so the two are directly comparable:
  * same dataset, features, correlation filter and discovery definition
    (a "discovery" = a material in the lowest DISCOVERY_PCT % of band gaps);
  * same random initial labeled set for every repeat (same seeds, same draw);
  * same experimental budget (N_ITER * BATCH picks after the initial set);
  * same repeats, same metrics, same output columns as sequential_learning_results.csv.

The ONLY difference: the model is trained ONCE on the initial labeled set and all
BUDGET picks are chosen in one shot from that single ranking (no retraining between batches).

Outputs (in OUT_DIR):
  static_learning_results.csv        one row per (model, strategy), SL-compatible columns
  static_learning_curves.csv         mean/std cumulative discoveries vs. materials measured
  static_curves_<model>.png          curves per model
  static_greedy_by_model.png         discovery yield of static greedy per model
  comparison_static_vs_sequential.* only if SL_RESULTS_CSV exists (see COMPARE_MODEL)
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor, BaggingRegressor)

# --------------------------------------------------------------------------
# Settings. Keep these identical to the SL run you are comparing against.
# --------------------------------------------------------------------------
RANDOM_STATE = 42
DISCOVERY_PCT = 5
N_INIT = 50
BATCH = 10
N_ITER = 30
N_REPEATS = 5
N_BOOT = 10                      # bootstrap models for uncertainty (only used by UCB)
UCB_LAMBDAS = [0.5, 1.0, 2.0]
BUDGET = N_ITER * BATCH          # total picks after the initial set (300)

OUT_DIR = "static_outputs"
FEATURE_CACHE = "features_cache.pkl"            # reuse the cache from the main script
SL_RESULTS_CSV = "sequential_learning_results.csv"   # SL table to compare against (optional)
COMPARE_MODEL = "GradientBoosting"   # must be in MODEL_NAMES; set to the model your SL run used

MODEL_NAMES = [
    "ElasticNet", "SVR", "GPR", "MLP",
    "RandomForest", "GradientBoosting", "Bagging"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_features():
    """Load + featurize matbench_expt_gap (cached after the first run)."""
    if os.path.exists(FEATURE_CACHE):
        print(f"Loading cached features from {FEATURE_CACHE}")
        return pd.read_pickle(FEATURE_CACHE)

    from matminer.datasets import load_dataset
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementProperty, Stoichiometry

    df = load_dataset("matbench_expt_gap")
    df = df.dropna(subset=["composition", "gap expt"])
    df = df.drop_duplicates(subset=["composition"]).reset_index(drop=True)
    df = df[df["gap expt"] > 0.0].reset_index(drop=True)

    df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
        df, "composition", ignore_errors=True, pbar=False)
    df = df.dropna(subset=["composition_obj"]).reset_index(drop=True)

    magpie = ElementProperty.from_preset("magpie")
    magpie.set_n_jobs(os.cpu_count() or 1)
    df_feat = magpie.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True, pbar=False)
    df_feat = Stoichiometry().featurize_dataframe(df_feat, col_id="composition_obj", ignore_errors=True, pbar=False)

    df_feat.to_pickle(FEATURE_CACHE)
    return df_feat


def prepare_xy(df_feat):
    """Same cleaning + |r| > 0.95 correlation filter as the main script."""
    X = df_feat.drop(columns=["composition", "composition_obj", "gap expt"], errors="ignore")
    y = df_feat["gap expt"].values

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] == 0:
        raise ValueError("All features were dropped before correlation filtering!")

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
    X = X.drop(columns=to_drop)
    return X.values, y, X.columns.tolist()


# --------------------------------------------------------------------------
# Models (fixed hyperparameters, no Optuna)
# --------------------------------------------------------------------------
def make_model(name):
    if name == "ElasticNet":
        return make_pipeline(StandardScaler(), ElasticNet(
            alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE))
    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05, gamma="scale"))
    if name == "GPR":
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=20.0, nu=2.5),
            alpha=1e-2, normalize_y=True, random_state=RANDOM_STATE))
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=1000,
            random_state=RANDOM_STATE))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=RANDOM_STATE)
    if name == "Bagging":
        return BaggingRegressor(
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def _final_step(model):
    return model[-1] if isinstance(model, Pipeline) else model


def predict_with_uncertainty(model, X_new, X_fit, y_fit, n_boot=N_BOOT, seed=0):
    """Mean and std per candidate. Free for GPR / tree ensembles, bootstrap otherwise."""
    final = _final_step(model)

    if isinstance(final, GaussianProcessRegressor):
        Xt = model[:-1].transform(X_new) if isinstance(model, Pipeline) else X_new
        mu, sigma = final.predict(Xt, return_std=True)
        return mu, sigma

    if isinstance(final, RandomForestRegressor):
        per_tree = np.array([t.predict(X_new) for t in final.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    if isinstance(final, BaggingRegressor):
        per_est = np.array([
            est.predict(X_new[:, feats])
            for est, feats in zip(final.estimators_, final.estimators_features_)])
        return per_est.mean(axis=0), per_est.std(axis=0)

    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X_fit), len(X_fit))
        preds.append(clone(model).fit(X_fit[idx], y_fit[idx]).predict(X_new))
    return model.predict(X_new), np.std(preds, axis=0)


# --------------------------------------------------------------------------
# Static (one-shot) screening
# --------------------------------------------------------------------------
def to_curve(hits):
    """Per-pick 0/1 discovery flags -> cumulative discoveries after every batch."""
    return np.cumsum(hits.reshape(N_ITER, BATCH).sum(axis=1))


def strategy_labels():
    return (["Random", "Static Greedy (λ=0)"]
            + [f"Static UCB (λ={lam})" for lam in UCB_LAMBDAS])


def run_static(template, X, y, seed):
    """One repeat: train once on the initial set, choose all BUDGET picks in one shot."""
    rng = np.random.default_rng(seed)
    is_top = y <= np.percentile(y, DISCOVERY_PCT)

    # Identical draw to the SL script -> identical initial labeled set for the same seed.
    labeled = rng.choice(np.where(~is_top)[0], N_INIT, replace=False)
    pool = np.setdiff1d(np.arange(len(y)), labeled)

    curves = {}
    random_pick = rng.choice(pool, BUDGET, replace=False)
    curves["Random"] = to_curve(is_top[random_pick])

    model = clone(template).fit(X[labeled], y[labeled])
    mu, sigma = predict_with_uncertainty(model, X[pool], X[labeled], y[labeled], seed=seed)

    for lam in [0.0] + UCB_LAMBDAS:
        score = -mu + lam * sigma                         # higher = more promising
        picks = pool[np.argsort(score)[::-1][:BUDGET]]    # one-shot ranking, no retraining
        label = "Static Greedy (λ=0)" if lam == 0 else f"Static UCB (λ={lam})"
        curves[label] = to_curve(is_top[picks])

    return curves, int(is_top.sum())


def summarize(curves_by_label, total_top):
    rows = []
    for label in strategy_labels():
        curves = np.array(curves_by_label[label])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        first_hit = [(np.argmax(c > 0) + 1) * BATCH if (c > 0).any() else np.nan for c in curves]
        rows.append({
            "Strategy": label,
            "Discoveries found": f"{mean[-1]:.1f} ± {std[-1]:.1f} / {total_top}",
            "Discovery Yield": curves[:, -1].mean() / BUDGET,
            "Discovery Probability": (curves[:, -1] > 0).mean(),
            "Experiments to 1st hit": (np.nan if np.all(np.isnan(first_hit)) else np.nanmean(first_hit)),
        })
    return pd.DataFrame(rows)


def run_model(name, X, y):
    template = make_model(name)
    by_label = {label: [] for label in strategy_labels()}
    total_top = None
    for r in range(N_REPEATS):
        curves, total_top = run_static(template, X, y, seed=RANDOM_STATE + r)
        for label, c in curves.items():
            by_label[label].append(c)
    return by_label, total_top


# --------------------------------------------------------------------------
# Plots / comparison
# --------------------------------------------------------------------------
def plot_curves(name, by_label):
    x = np.arange(1, N_ITER + 1) * BATCH
    plt.figure(figsize=(8, 6))
    for label in strategy_labels():
        c = np.array(by_label[label])
        m, s = c.mean(axis=0), c.std(axis=0)
        plt.plot(x, m, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.15)
    plt.xlabel("Materials measured (after initial set)")
    plt.ylabel(f"Cumulative discoveries (best {DISCOVERY_PCT}%)")
    plt.title(f"Static (one-shot) screening - {name} ({N_REPEATS} repeats)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"static_curves_{name}.png"))
    plt.close()


def plot_greedy_by_model(results):
    g = results[results["Strategy"] == "Static Greedy (λ=0)"].sort_values("Discovery Yield")
    plt.figure(figsize=(8, 5))
    plt.barh(g["Model"], g["Discovery Yield"], color="teal")
    plt.xlabel("Discovery yield (discoveries / measured)")
    plt.title("Static greedy screening: discovery yield by model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "static_greedy_by_model.png"))
    plt.close()


def compare_with_sequential(results):
    """Merge static results for COMPARE_MODEL with the SL table, if the SL csv is present."""
    if not os.path.exists(SL_RESULTS_CSV):
        print(f"\n(No {SL_RESULTS_CSV} found - skipping the static-vs-sequential table. "
              f"Put the SL csv next to this script to get it.)")
        return None

    sl = pd.read_csv(SL_RESULTS_CSV)
    sl.insert(0, "Type", "Sequential")
    st = results[results["Model"] == COMPARE_MODEL].drop(columns="Model").copy()
    st.insert(0, "Type", "Static (one-shot)")
    both = pd.concat([sl, st], ignore_index=True)

    both.to_csv(os.path.join(OUT_DIR, "comparison_static_vs_sequential.csv"), index=False)
    try:
        both.to_excel(os.path.join(OUT_DIR, "comparison_static_vs_sequential.xlsx"), index=False)
    except Exception as e:      # openpyxl missing etc.
        print(f"  (xlsx not written: {e})")

    both["_mean"] = both["Discoveries found"].str.split("±").str[0].astype(float)
    plt.figure(figsize=(9, 6))
    colors = ["tab:blue" if t == "Sequential" else "tab:orange" for t in both["Type"]]
    plt.barh(both["Type"] + " | " + both["Strategy"], both["_mean"], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Mean discoveries found in {BUDGET} measurements")
    plt.title(f"Sequential vs static screening ({COMPARE_MODEL})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_static_vs_sequential.png"))
    plt.close()
    return both.drop(columns="_mean")


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y, _ = prepare_xy(load_features())
    print(f"Dataset: {X.shape[0]} materials, {X.shape[1]} features | budget = {BUDGET} picks "
          f"after {N_INIT} initial, {N_REPEATS} repeats")

    all_rows, curve_rows = [], []
    for name in MODEL_NAMES:
        t0 = time.time()
        try:
            by_label, total_top = run_model(name, X, y)
        except Exception as e:
            print(f"  x {name:<18} skipped ({type(e).__name__}: {e})")
            continue
        df = summarize(by_label, total_top)
        df.insert(0, "Model", name)
        all_rows.append(df)

        x = np.arange(1, N_ITER + 1) * BATCH
        for label in strategy_labels():
            c = np.array(by_label[label])
            for xi, m, s in zip(x, c.mean(axis=0), c.std(axis=0)):
                curve_rows.append({"Model": name, "Strategy": label, "Measured": xi, "Mean": m, "Std": s})
        plot_curves(name, by_label)
        print(f"  ok {name:<18} done in {time.time() - t0:.1f}s")

    if not all_rows:
        raise RuntimeError("No model could be run.")

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(os.path.join(OUT_DIR, "static_learning_results.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(OUT_DIR, "static_learning_curves.csv"), index=False)
    plot_greedy_by_model(results)

    print("\n" + "=" * 78)
    print("             STATIC (NON-SEQUENTIAL) SCREENING PERFORMANCE")
    print("=" * 78)
    print(results.round(4).to_string(index=False))

    both = compare_with_sequential(results)
    if both is not None:
        print("\n" + "=" * 78)
        print(f"        SEQUENTIAL vs STATIC  (static model = {COMPARE_MODEL})")
        print("=" * 78)
        print(both.round(4).to_string(index=False))

    print(f"\nAll outputs saved in ./{OUT_DIR}/   (total {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
