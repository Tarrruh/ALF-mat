import warnings
warnings.filterwarnings("ignore")
import time
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import shap
from joblib import Parallel, delayed

from matminer.datasets import load_dataset
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty, Stoichiometry

from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    BaggingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42

FAST = True

if FAST:
    N_TRIALS = 10
    TIMEOUT_PER_MODEL = 60       
    TUNE_MAX_N = 1500            
    AL_N_ITER = 15
    AL_N_REPEATS = 3
    AL_N_BOOT = 5
    SHAP_N_TREE = 100
    SHAP_N_KERNEL = 25
else:
    N_TRIALS = 25
    TIMEOUT_PER_MODEL = 180
    TUNE_MAX_N = 2500
    AL_N_ITER = 30
    AL_N_REPEATS = 5
    AL_N_BOOT = 10
    SHAP_N_TREE = 200
    SHAP_N_KERNEL = 40

CV_FOLDS = 5
DISCOVERY_PCT = 5
AL_N_INIT = 50
AL_BATCH = 10
UCB_LAMBDAS = [0.5, 1.0, 2.0]
AL_N_JOBS = -1                  
FEATURE_CACHE = "features_cache.pkl"   

INNER_PARALLEL = {"RandomForest", "Bagging"}
TREE_SHAP = {"RandomForest", "GradientBoosting"}

MODEL_NAMES = [
    "ElasticNet", "SVR", "GPR", "MLP",
    "RandomForest", "GradientBoosting", "Bagging"]


@contextmanager
def stage(label):
    t0 = time.time()
    print(f"[{label}] started...")
    yield
    print(f"[{label}] done in {time.time() - t0:.1f}s")


def calculate_discovery_metrics(y_true, y_pred, k=30, higher_is_better=False):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if higher_is_better:
        pred_top_idx = np.argsort(y_pred)[-k:]
        true_top_idx = np.argsort(y_true)[-k:]
        threshold = np.percentile(y_true, 95)
        discoveries = np.sum(y_true[pred_top_idx] >= threshold)
    else:
        pred_top_idx = np.argsort(y_pred)[:k]
        true_top_idx = np.argsort(y_true)[:k]
        threshold = np.percentile(y_true, 5)
        discoveries = np.sum(y_true[pred_top_idx] <= threshold)

    dy = discoveries / k
    dp = 1 if discoveries > 0 else 0
    p_at_k = len(set(pred_top_idx) & set(true_top_idx)) / k

    return dy, dp, p_at_k


def evaluate(y_true, y_pred, k):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dy, dp, p_at_k = calculate_discovery_metrics(y_true, y_pred, k=k, higher_is_better=False)
    return dict(RMSE=rmse, MAE=mae, R2=r2, DY=dy, DP=dp, P_at_k=p_at_k)


def build_model(name, trial):
    s = trial.suggest_float
    si = trial.suggest_int

    if name == "ElasticNet":
        return make_pipeline(StandardScaler(), ElasticNet(
            alpha=s("alpha", 1e-4, 1.0, log=True), l1_ratio=s("l1_ratio", 0.05, 0.95),
            max_iter=5000, random_state=RANDOM_STATE))

    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(
            C=s("C", 1e-1, 1e2, log=True), epsilon=s("epsilon", 1e-3, 0.5, log=True),
            gamma=s("gamma", 1e-4, 1e-1, log=True)))

    if name == "GPR":
        length_scale = s("length_scale", 1.0, 200.0, log=True)
        amplitude = s("amplitude", 0.1, 10.0, log=True)
        kt = trial.suggest_categorical("kernel", ["rbf", "matern15", "matern25"])
        base = RBF(length_scale) if kt == "rbf" else Matern(length_scale, nu=1.5 if kt == "matern15" else 2.5)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel(amplitude) * base, alpha=s("noise", 1e-6, 1e-1, log=True),
            normalize_y=True, optimizer=None))

    if name == "MLP":
        n_layers = si("n_layers", 1, 3)
        hidden = tuple(si(f"units_{i}", 32, 128, log=True) for i in range(n_layers))
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=hidden, alpha=s("alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=s("lr", 1e-4, 1e-2, log=True), early_stopping=True,
            n_iter_no_change=10, max_iter=300, random_state=RANDOM_STATE))

    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=si("n_estimators", 100, 300), max_depth=si("max_depth", 5, 40),
            min_samples_leaf=si("min_samples_leaf", 1, 5), max_features=s("max_features", 0.2, 1.0),
            random_state=RANDOM_STATE, n_jobs=-1)

    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=si("n_estimators", 100, 300), learning_rate=s("learning_rate", 0.02, 0.3, log=True),
            max_depth=si("max_depth", 2, 6), subsample=s("subsample", 0.5, 1.0),
            min_samples_leaf=si("min_samples_leaf", 1, 10), max_features=s("max_features", 0.3, 1.0),
            random_state=RANDOM_STATE)

    if name == "Bagging":
        return BaggingRegressor(
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            n_estimators=si("n_estimators", 50, 200), max_samples=s("max_samples", 0.5, 1.0),
            max_features=s("max_features", 0.3, 1.0), random_state=RANDOM_STATE, n_jobs=-1)

    raise ValueError(f"Unknown model: {name}")


def tune_model(name, X, y):
    if TUNE_MAX_N and len(y) > TUNE_MAX_N:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y), TUNE_MAX_N, replace=False)
        X, y = X[idx], y[idx]

    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_jobs = 1 if name in INNER_PARALLEL else -1

    def objective(trial):
        model = build_model(name, trial)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=cv_jobs)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_PER_MODEL, catch=(Exception,))

    template = build_model(name, optuna.trial.FixedTrial(study.best_params))
    return template, study.best_value, study.best_params


def _final_step(model):
    return model[-1] if isinstance(model, Pipeline) else model


def _single_thread(model):
    """Clone a model and force n_jobs=1 so parallel AL workers don't oversubscribe the CPU."""
    m = clone(model)
    final = _final_step(m)
    if "n_jobs" in final.get_params():
        final.set_params(n_jobs=1)
    return m


def predict_with_uncertainty(model, X_new, X_fit=None, y_fit=None, n_boot=AL_N_BOOT, seed=0):
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
        m = clone(model).fit(X_fit[idx], y_fit[idx])
        preds.append(m.predict(X_new))
    return model.predict(X_new), np.std(preds, axis=0)


def run_sequential_learning(template, X, y, lam, random_selection, seed):
    template = _single_thread(template)
    rng = np.random.default_rng(seed)
    n = len(y)
    is_top = y <= np.percentile(y, DISCOVERY_PCT)

    labeled = rng.choice(np.where(~is_top)[0], AL_N_INIT, replace=False).tolist()
    in_pool = np.ones(n, dtype=bool)
    in_pool[labeled] = False

    n_found, curve = 0, []
    for it in range(AL_N_ITER):
        pool_idx = np.where(in_pool)[0]
        if random_selection:
            pick = rng.choice(pool_idx, AL_BATCH, replace=False)
        else:
            model = clone(template).fit(X[labeled], y[labeled])
            if lam == 0:
                score = -model.predict(X[pool_idx])
            else:
                mu, sigma = predict_with_uncertainty(
                    model, X[pool_idx], X[labeled], y[labeled], seed=seed + it)
                score = -mu + lam * sigma
            pick = pool_idx[np.argsort(score)[-AL_BATCH:]]

        in_pool[pick] = False
        labeled.extend(pick.tolist())
        n_found += int(is_top[pick].sum())
        curve.append(n_found)

    return np.array(curve), int(is_top.sum())


def compare_acquisition_strategies(template, X, y, model_name):
    strategies = [("Random", None, True), ("Greedy (λ=0)", 0.0, False)]
    strategies += [(f"UCB (λ={l})", l, False) for l in UCB_LAMBDAS]

    x_axis = np.arange(1, AL_N_ITER + 1) * AL_BATCH
    n_acquired = AL_N_ITER * AL_BATCH

    jobs = [(label, lam, is_random, r)
            for label, lam, is_random in strategies
            for r in range(AL_N_REPEATS)]
    outputs = Parallel(n_jobs=AL_N_JOBS)(
        delayed(run_sequential_learning)(template, X, y, lam, is_random, RANDOM_STATE + r)
        for _, lam, is_random, r in jobs)

    by_label = {}
    total_top = None
    for (label, _, _, _), (curve, tt) in zip(jobs, outputs):
        by_label.setdefault(label, []).append(curve)
        total_top = tt

    rows = []
    plt.figure(figsize=(8, 6))
    for label, _, _ in strategies:
        curves = np.array(by_label[label])
        mean, std = curves.mean(axis=0), curves.std(axis=0)

        first_hit = [(np.argmax(c > 0) + 1) * AL_BATCH if (c > 0).any() else np.nan for c in curves]
        rows.append({
            "Strategy": label,
            "Discoveries found": f"{mean[-1]:.1f} ± {std[-1]:.1f} / {total_top}",
            "Discovery Yield": curves[:, -1].mean() / n_acquired,
            "Discovery Probability": (curves[:, -1] > 0).mean(),
            "Experiments to 1st hit": np.nanmean(first_hit) if not np.all(np.isnan(first_hit)) else np.nan,
        })
        plt.plot(x_axis, mean, label=label)
        plt.fill_between(x_axis, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Materials measured (after initial set)")
    plt.ylabel(f"Cumulative discoveries (best {DISCOVERY_PCT}%)")
    plt.title(f"Sequential learning – {model_name} ({AL_N_REPEATS} repeats)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sequential_learning.png")
    plt.close()

    return pd.DataFrame(rows)


def explain_with_shap(model, model_name, X_train, X_test, feature_names):
    rng = np.random.default_rng(RANDOM_STATE)
    is_tree = model_name in TREE_SHAP
    n = min(SHAP_N_TREE if is_tree else SHAP_N_KERNEL, len(X_test))
    idx = rng.choice(len(X_test), n, replace=False)
    X_exp = X_test[idx]

    if isinstance(model, Pipeline):
        transform, final = model[:-1].transform, model[-1]
    else:
        transform, final = (lambda a: a), model

    if is_tree:
        explainer = shap.TreeExplainer(final)
        sv = explainer.shap_values(transform(X_exp))
    else:
        background = shap.kmeans(transform(X_train), 10)
        explainer = shap.KernelExplainer(final.predict, background)
        sv = explainer.shap_values(transform(X_exp), nsamples=150)

    sv = np.array(sv)
    X_exp_df = pd.DataFrame(X_exp, columns=feature_names)

    shap.summary_plot(sv, X_exp_df, max_display=15, show=False)
    plt.title(f"SHAP summary – {model_name}")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    shap.summary_plot(sv, X_exp_df, plot_type="bar", max_display=15, show=False)
    plt.title("Mean |SHAP| (eV)")
    plt.tight_layout()
    plt.savefig("shap_importance_bar.png")
    plt.close()

    preds = model.predict(X_exp)
    j = int(np.argmin(preds))
    base = float(np.ravel(explainer.expected_value)[0])
    shap.plots.waterfall(
        shap.Explanation(values=sv[j], base_values=base, data=X_exp[j], feature_names=feature_names),
        max_display=12, show=False)
    plt.title(f"Local explanation – lowest predicted gap ({preds[j]:.3f} eV)")
    plt.tight_layout()
    plt.savefig("shap_waterfall.png")
    plt.close()

    importance = pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)
    return importance


def save_correlation_heatmap(corr, title, filename, annotate=False):
    size = 10 if annotate else 8
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    if annotate:
        labels = [str(c).replace("MagpieData ", "") for c in corr.columns]
        n = len(labels)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


def load_features():
    """Load + featurize the dataset, caching the result so it only runs once."""
    if os.path.exists(FEATURE_CACHE):
        print(f"Loading cached features from {FEATURE_CACHE}")
        return pd.read_pickle(FEATURE_CACHE)

    n_cpu = os.cpu_count() or 1
    df = load_dataset("matbench_expt_gap")
    df = df.dropna(subset=["composition", "gap expt"])
    df = df.drop_duplicates(subset=["composition"]).reset_index(drop=True)
    df = df[df["gap expt"] > 0.0].reset_index(drop=True)

    str_to_comp = StrToComposition(target_col_id="composition_obj")
    df = str_to_comp.featurize_dataframe(df, "composition", ignore_errors=True, pbar=False)
    df = df.dropna(subset=["composition_obj"]).reset_index(drop=True)

    magpie = ElementProperty.from_preset("magpie")
    magpie.set_n_jobs(n_cpu)
    df_feat = magpie.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True, pbar=False)

    stoich = Stoichiometry()
    df_feat = stoich.featurize_dataframe(df_feat, col_id="composition_obj", ignore_errors=True, pbar=False)

    df_feat.to_pickle(FEATURE_CACHE)
    return df_feat


if __name__ == '__main__':
    t_start = time.time()
    print(f"Mode: {'FAST' if FAST else 'FULL'}")

    with stage("Load + featurize"):
        df_feat = load_features()

    X = df_feat.drop(columns=["composition", "composition_obj", "gap expt"], errors="ignore")
    y = df_feat["gap expt"].values

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median())
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] == 0:
        raise ValueError("All features were dropped before correlation filtering!")

    with stage("Correlation filtering"):
        corr_full = X.corr()                      
        save_correlation_heatmap(
            corr_full, f"Feature correlation matrix before filtering ({X.shape[1]} features)", "corr_before.png")

        upper = corr_full.abs().where(np.triu(np.ones(corr_full.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] > 0.95).any()]
        keep = [c for c in X.columns if c not in set(to_drop)]
        X = X[keep]

        save_correlation_heatmap(
            corr_full.loc[keep, keep],
            f"Feature correlation matrix after removing |r| > 0.95 ({X.shape[1]} features)", "corr_after.png")

    feature_names = X.columns.tolist()
    X_all = X.values

    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.20, random_state=42)
    K = min(50, len(y_test))

    print(f"\nTuning {len(MODEL_NAMES)} models with Optuna "
          f"({N_TRIALS} trials, {CV_FOLDS}-fold CV, tune subsample={TUNE_MAX_N})...")
    results, fitted, templates = [], {}, {}

    for name in MODEL_NAMES:
        t0 = time.time()
        try:
            template, cv_rmse, best_params = tune_model(name, X_train, y_train)
            model = clone(template).fit(X_train, y_train)
            m = evaluate(y_test, model.predict(X_test), K)
            results.append({"Model": name, "CV_RMSE": cv_rmse, **m, "Time_s": time.time() - t0})
            fitted[name], templates[name] = model, template
            print(f"  ✓ {name:<18} completed in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ✗ {name:<18} skipped ({type(e).__name__}: {e})")
            continue

    if not results:
        raise RuntimeError("No model could be trained.")

    results_df = pd.DataFrame(results).sort_values("CV_RMSE").reset_index(drop=True)
    best_name = results_df.loc[0, "Model"]
    best_model, best_template = fitted[best_name], templates[best_name]
    print(f"\nBest model by CV RMSE: {best_name}")

    plt.figure(figsize=(8, 5))
    plt.barh(results_df["Model"][::-1], results_df["CV_RMSE"][::-1], color="teal")
    plt.xlabel("CV RMSE (eV)")
    plt.title("Model comparison (Optuna-tuned, lower is better)")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

    y_pred = best_model.predict(X_test)
    final = evaluate(y_test, y_pred, K)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="navy")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="crimson")
    plt.xlabel("Actual Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.title(f"{best_name}: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    plt.close()

    with stage("Uncertainty on test set"):
        _, uncertainty = predict_with_uncertainty(best_model, X_test, X_train, y_train)
    absolute_error = np.abs(y_test - y_pred)

    plt.figure(figsize=(7, 5))
    plt.scatter(uncertainty, absolute_error, alpha=0.6, color="purple")
    plt.xlabel("Prediction Uncertainty (eV)")
    plt.ylabel("Absolute Prediction Error (eV)")
    plt.title(f"Uncertainty vs Prediction Error ({best_name})")
    plt.tight_layout()
    plt.savefig("uncertainty_vs_error.png")
    plt.close()

    with stage("Sequential learning"):
        al_df = compare_acquisition_strategies(best_template, X_all, y, best_name)

    with stage("SHAP"):
        importance = explain_with_shap(best_model, best_name, X_train, X_test, feature_names)

    top_features = importance.head(15).index.tolist()
    corr_df = pd.DataFrame(X_all, columns=feature_names)[top_features].copy()
    corr_df["Band gap (eV)"] = y
    save_correlation_heatmap(corr_df.corr(), "Top SHAP features and band gap: correlation matrix", "corr_shap.png", annotate=True)

    print("\n" + "=" * 70)
    print("                     MODEL BENCHMARK COMPARISON                      ")
    print("=" * 70)
    print(results_df[["Model", "CV_RMSE", "RMSE", "MAE", "R2", "P_at_k", "Time_s"]].round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"               WINNING MODEL: {best_name.upper()} DETAILS               ")
    print("=" * 70)
    print(f"Test RMSE              : {final['RMSE']:.4f} eV")
    print(f"Test MAE               : {final['MAE']:.4f} eV")
    print(f"R² Score               : {final['R2']:.4f}")
    print(f"Discovery Yield (DY)   : {final['DY']:.4f}")
    print(f"Discovery Prob. (DP)   : {final['DP']}")
    print(f"Precision@{K}           : {final['P_at_k']:.4f}")

    print("\nSequential Learning Performance:")
    print(al_df.round(4).to_string(index=False))

    print("\nTop 10 Feature Importances (|SHAP|):")
    print(importance.head(10).round(4).to_string())
    print("\n" + "=" * 70)
    print("All figures have been saved to the current working directory as .png files.")
    print(f"Total runtime: {time.time() - t_start:.1f}s")
    print("=" * 70)