"""
compare_models.py
--------------------
Time-series rolling/expanding CV with enhanced feature engineering and model comparison.

Features:
- Expanding-window CV (chronological) with K folds (default 4)
- Leak-free feature engineering from A..N (+ time):
  * Lags: [1,2,5,10,20,60,120,240]
  * Rolling mean/std on raw A..N and first differences: windows [5,20,60,120] (shifted by 1)
  * First differences and returns (shifted)
  * Rolling correlations between top-variance base features (K=6) over windows [20,60,120] (shifted)
- Optional PCA compression (train-fold fit, test-fold transform) replacing engineered features
- Models: LightGBM, XGBoost, CatBoost for Y1 and Y2 (with different regularization)
- Simple blend: (XGBoost + CatBoost)/2
- Outputs per-fold and mean R^2; saves CSV

Usage:
    python compare_models.py --csv ./data/train.csv --out ./results --folds 4 --use_pca 0 --pca_components 32
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Optional models (script skips those not installed)
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


# -----------------------------
# Feature Engineering (leak-free)
# -----------------------------
def _pick_top_variance_features(df, base_cols, k=6):
    variances = df[base_cols].var().sort_values(ascending=False)
    return variances.index[:min(k, len(variances))].tolist()

def make_features_full(df,
                       lag_list=(1,2,5,10,20,60,120,240),
                       roll_windows=(5,20,60,120),
                       rcorr_windows=(20,60,120),
                       rcorr_topk=6):
    """Create engineered features from A..N and time only (no target leakage).
       All rolling/lagged stats are shifted to exclude current observation."""
    df = df.copy()
    # Identify base numeric features A..N (exclude time, Y1, Y2)
    base_cols = [c for c in df.columns if c not in ['time','Y1','Y2']]

    # Lags
    for lag in lag_list:
        for c in base_cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    # First differences and simple returns (shifted so they end at t-1)
    for c in base_cols:
        df[f"{c}_diff1"] = df[c].diff().shift(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = (df[c] / df[c].shift(1)) - 1.0
        df[f"{c}_ret1"] = ret.shift(1)

    # Rolling mean/std on raw and diffs (shifted)
    for w in roll_windows:
        for c in base_cols:
            df[f"{c}_rollmean{w}"] = df[c].shift(1).rolling(w).mean()
            df[f"{c}_rollstd{w}"]  = df[c].shift(1).rolling(w).std()
            # On differences
            dcol = f"{c}_diff1"
            df[f"{dcol}_rollmean{w}"] = df[dcol].rolling(w).mean()
            df[f"{dcol}_rollstd{w}"]  = df[dcol].rolling(w).std()

    # Rolling correlations between top-variance base features
    topv = _pick_top_variance_features(df, base_cols, k=rcorr_topk)
    for w in rcorr_windows:
        for i in range(len(topv)):
            for j in range(i+1, len(topv)):
                a, b = topv[i], topv[j]
                # Use shifted series to avoid leakage
                s1 = df[a].shift(1)
                s2 = df[b].shift(1)
                df[f"rcorr_{a}_{b}_w{w}"] = s1.rolling(w).corr(s2)

    # Drop rows with NaNs created by lags/rolling
    df = df.dropna().reset_index(drop=True)

    # Collect feature columns (keep 'time' for splitting)
    feature_cols = [c for c in df.columns if c not in ['Y1','Y2']]
    X_all = df[feature_cols]
    y1_all = df['Y1']
    y2_all = df['Y2']
    return X_all, y1_all, y2_all


# -----------------------------
# Time-series expanding CV
# -----------------------------
def build_folds_by_time(times, n_folds=4, min_train_frac=0.5):
    """
    Expanding-window folds. Always train on all data up to fold start, validate on the next slice.
    times: numpy array (same length as X), containing time stamps (numeric or sortable)
    Returns: list of (train_idx, val_idx)
    """
    uniq = np.unique(times)
    n = len(uniq)
    if n_folds < 2:
        n_folds = 2
    # define fold cut points
    # ensure the first train covers at least min_train_frac of time range
    start_train_idx = int(n * min_train_frac)
    # remaining is split into n_folds equal-ish segments
    remain = n - start_train_idx
    seg = max(1, remain // n_folds)

    folds = []
    for k in range(n_folds):
        val_start = start_train_idx + k * seg
        val_end = start_train_idx + (k + 1) * seg if k < n_folds - 1 else n
        if val_start >= val_end or val_start >= n:
            continue
        train_time_max = uniq[val_start - 1]  # train up to just before val
        val_time_start = uniq[val_start]
        val_time_end = uniq[val_end - 1]

        train_idx = np.where(times <= train_time_max)[0]
        val_mask = (times >= val_time_start) & (times <= val_time_end)
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


# -----------------------------
# Models & params
# -----------------------------
def build_models_y1():
    models = []
    if LGBMRegressor is not None:
        models.append(("LightGBM",
            LGBMRegressor(objective='regression',
                          learning_rate=0.03,
                          n_estimators=2000,
                          num_leaves=63,
                          min_data_in_leaf=300,
                          subsample=0.9,
                          colsample_bytree=0.9,
                          random_state=42)))
    if XGBRegressor is not None:
        models.append(("XGBoost",
            XGBRegressor(objective='reg:squarederror',
                         learning_rate=0.02,
                         n_estimators=3500,
                         max_depth=5,
                         subsample=0.9,
                         colsample_bytree=0.9,
                         reg_lambda=3.0,
                         tree_method='hist',
                         random_state=42,
                         verbosity=0)))
    if CatBoostRegressor is not None:
        models.append(("CatBoost",
            CatBoostRegressor(loss_function='RMSE',
                              learning_rate=0.03,
                              depth=6,
                              l2_leaf_reg=8.0,
                              iterations=3000,
                              random_seed=42,
                              verbose=False)))
    return models

def build_models_y2():
    models = []
    if LGBMRegressor is not None:
        models.append(("LightGBM",
            LGBMRegressor(objective='regression',
                          learning_rate=0.03,
                          n_estimators=2500,
                          num_leaves=127,
                          min_data_in_leaf=200,   # allow more splits
                          subsample=0.85,
                          colsample_bytree=0.85,
                          lambda_l2=2.0,
                          random_state=42)))
    if XGBRegressor is not None:
        models.append(("XGBoost",
            XGBRegressor(objective='reg:squarederror',
                         learning_rate=0.02,
                         n_estimators=4000,
                         max_depth=6,
                         subsample=0.85,
                         colsample_bytree=0.85,
                         reg_lambda=4.0,
                         tree_method='hist',
                         random_state=42,
                         verbosity=0)))
    if CatBoostRegressor is not None:
        models.append(("CatBoost",
            CatBoostRegressor(loss_function='RMSE',
                              learning_rate=0.03,
                              depth=7,
                              l2_leaf_reg=10.0,
                              iterations=3500,
                              random_seed=42,
                              verbose=False)))
    return models


# -----------------------------
# CV Runner
# -----------------------------
def run_cv(X, y, models, folds, use_pca=False, pca_components=32):
    """
    For each fold:
      - fit (optional) scaler+PCA on X_train
      - train each model, score on val (R^2)
    Returns dict: {model_name: [fold_r2, ...]}, and (optional) per-fold pred cache
    """
    results = {name: [] for name, _ in models}
    # We'll also store per-fold predictions for blending
    preds_cache = {name: [] for name, _ in models}

    for fold_id, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_va = y.iloc[va_idx].copy()

        # Remove time column from features
        feat_cols = [c for c in X_tr.columns if c != 'time']
        X_tr_f = X_tr[feat_cols].values
        X_va_f = X_va[feat_cols].values

        # Optional PCA (fit on train fold only)
        if use_pca:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr_s = scaler.fit_transform(X_tr_f)
            X_va_s = scaler.transform(X_va_f)
            pca = PCA(n_components=min(pca_components, X_tr_s.shape[1]))
            X_tr_use = pca.fit_transform(X_tr_s)
            X_va_use = pca.transform(X_va_s)
        else:
            X_tr_use = X_tr_f
            X_va_use = X_va_f

        for name, model in models:
            # Fit and predict
            model.fit(X_tr_use, y_tr)
            y_hat = model.predict(X_va_use)
            r2 = r2_score(y_va, y_hat)
            results[name].append(r2)
            preds_cache[name].append((va_idx, y_hat))

        print(f"[fold {fold_id}/{len(folds)}] done")

    return results, preds_cache


def blend_from_cache(preds_cache, y, folds, blend_names=("XGBoost","CatBoost"), blend_label="Blend_XGB_CAT"):
    """Create a simple average blend of listed models' predictions and compute per-fold R^2."""
    if not all(name in preds_cache for name in blend_names):
        return None  # some model missing
    blend_scores = []
    for (tr_idx, va_idx) in folds:
        # Collect matching fold preds
        fold_preds = []
        for name in blend_names:
            fold_list = preds_cache[name]
            # find tuple with the same va_idx (by identity of indices)
            # assuming same folds order, we can just align by position:
            # get the next stored y_hat for this fold
        # Safer: rely on order
        pass
    # Since reconstructing by indices is verbose, simpler approach:
    # assume aligned order of fold appends; re-implement blend using order.

def blend_by_order(preds_cache, y, folds, blend_names=("XGBoost","CatBoost"), blend_label="Blend_XGB_CAT"):
    if not all(name in preds_cache for name in blend_names):
        return None, None
    # preds_cache[name] is list of (va_idx, y_hat) in fold order
    scores = []
    blend_preds_cache = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        # Fetch in same fold order
        yhats = []
        for name in blend_names:
            idxs, pred = preds_cache[name][fold_i]
            # Sanity: ensure val indices align
            if not np.array_equal(idxs, va_idx):
                # If not aligned, skip blending
                return None, None
            yhats.append(pred)
        # Average
        y_blend = np.mean(np.vstack(yhats), axis=0)
        r2 = r2_score(y.iloc[va_idx], y_blend)
        scores.append(r2)
        blend_preds_cache.append((va_idx, y_blend))
    return {blend_label: scores}, {blend_label: blend_preds_cache}


# -----------------------------
# Main
# -----------------------------
def main(csv_path: str, out_dir: str, n_folds: int, use_pca: int, pca_components: int):
    os.makedirs(out_dir, exist_ok=True)

    # Load & sort chronologically
    df = pd.read_csv(csv_path).sort_values('time').reset_index(drop=True)

    # Build features
    X_all, y1_all, y2_all = make_features_full(df)

    # Build expanding-window folds (by time)
    times = X_all['time'].values
    folds = build_folds_by_time(times, n_folds=n_folds, min_train_frac=0.5)
    if len(folds) == 0:
        raise RuntimeError("No CV folds created. Try reducing min_train_frac or the number of folds.")

    # Announce model availability
    available = [
        ("LightGBM", LGBMRegressor is not None),
        ("XGBoost", XGBRegressor is not None),
        ("CatBoost", CatBoostRegressor is not None),
    ]
    print("\n[model availability]")
    for name, ok in available:
        print(f" - {name}: {'OK' if ok else 'NOT FOUND'}")

    # -------- Y1 --------
    print("\n[Y1] Running expanding-window CV...")
    models_y1 = build_models_y1()
    res_y1, cache_y1 = run_cv(X_all, y1_all, models_y1, folds, use_pca=bool(use_pca), pca_components=pca_components)

    # Blend (XGB + CatBoost) if both present
    blend_scores_y1, blend_cache_y1 = blend_by_order(cache_y1, y1_all, folds, ("XGBoost","CatBoost"), "Blend_XGB_CAT")
    if blend_scores_y1 is not None:
        res_y1.update(blend_scores_y1)

    # -------- Y2 --------
    print("\n[Y2] Running expanding-window CV...")
    models_y2 = build_models_y2()
    res_y2, cache_y2 = run_cv(X_all, y2_all, models_y2, folds, use_pca=bool(use_pca), pca_components=pca_components)

    # Blend (XGB + CatBoost) if both present
    blend_scores_y2, blend_cache_y2 = blend_by_order(cache_y2, y2_all, folds, ("XGBoost","CatBoost"), "Blend_XGB_CAT")
    if blend_scores_y2 is not None:
        res_y2.update(blend_scores_y2)

    # Summaries
    rows = []
    for name, scores in res_y1.items():
        rows.append({"target": "Y1", "model": name, "fold_mean_r2": float(np.mean(scores)), "fold_std_r2": float(np.std(scores)), "folds": scores})
    for name, scores in res_y2.items():
        rows.append({"target": "Y2", "model": name, "fold_mean_r2": float(np.mean(scores)), "fold_std_r2": float(np.std(scores)), "folds": scores})

    out_df = pd.DataFrame(rows).sort_values(["target","fold_mean_r2"], ascending=[True, False])
    print("\n=== Expanding-window CV: mean R^2 over folds ===")
    print(out_df[["target","model","fold_mean_r2","fold_std_r2"]].to_string(index=False))

    # Save CSV + a verbose JSON-like column with per-fold scores
    out_path = os.path.join(out_dir, "model_r2_cv_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved CV summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--out", type=str, default="./results", help="Output directory")
    parser.add_argument("--folds", type=int, default=4, help="Number of expanding-window folds (3–5 recommended)")
    parser.add_argument("--use_pca", type=int, default=0, help="Set 1 to use PCA compression (fit per-fold on train)")
    parser.add_argument("--pca_components", type=int, default=32, help="PCA components if --use_pca=1")
    args = parser.parse_args()
    main(args.csv, args.out, args.folds, args.use_pca, args.pca_components)

# python compare_models.py --csv ./data/train.csv --out ./results --folds 4 --use_pca 0 --pca_components 32