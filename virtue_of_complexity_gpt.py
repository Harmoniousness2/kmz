"""
Updated: Fixed NaN/infinite handling during demo and improved data alignment.

Changes made:
- `standardize_returns`: now uses `min_periods=window` to produce NaNs for the initial periods (avoids divide-by-zero / infinities).
- `run_experiment` and `sweep_experiment`: explicitly align `returns` and `X_raw` with `dropna()` before generating features so S and returns contain no NaNs when passed to fast routines.
- `main_cli` non-sweep branch: now calls `run_experiment` for correct alignment (instead of calling fast_rolling_predictions directly on unaligned arrays).
- Added guards so `fast_rolling_predictions` returns NaN predictions if there are insufficient observations.

This resolves the RuntimeWarning you reported which was caused by NaNs / infinities in the standardized returns (division by zero in early rolling vol) leading to NaNs in the sufficient statistics matrices.
"""

from __future__ import annotations
import argparse
import math
from typing import Sequence, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


# ------------------------- IO & preprocessing -------------------------

def load_returns(csv_path: str, date_col: Optional[str] = None, return_col: str = "return") -> pd.Series:
    df = pd.read_csv(csv_path)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    if return_col not in df.columns:
        raise ValueError(f"Return column '{return_col}' not found in CSV")
    s = pd.Series(df[return_col].astype(float), index=df.index if date_col is not None else None, name="return")
    return s


def standardize_returns(returns: pd.Series, window: int = 12, clip_quantile: Optional[float] = None) -> pd.Series:
    """Standardize returns by trailing standard deviation over `window` periods.

    To avoid division-by-zero or spurious infinities the rolling std uses min_periods=window
    so the first `window-1` values will be NaN and can be dropped by the alignment step.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    vol = returns.rolling(window=window, min_periods=window).std(ddof=0)
    if clip_quantile is not None:
        qwin = max(window * 3, 21)
        floor = vol.rolling(window=qwin, min_periods=1).quantile(clip_quantile)
        vol = vol.clip(lower=floor.fillna(method="bfill").fillna(method="ffill"))
    standardized = returns / vol
    return standardized


# ------------------------- Random Fourier Features -------------------------

def generate_random_fourier_features(X: np.ndarray, L: int, gammas: Sequence[float], seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, d = X.shape
    m = len(gammas)
    k = math.ceil(L / (2 * m))
    features_list = []
    for gamma in gammas:
        omegas = rng.normal(size=(k, d)) / math.sqrt(gamma)
        projected = X @ omegas.T
        sin_feats = np.sin(projected)
        cos_feats = np.cos(projected)
        combined = np.empty((T, 2 * k))
        combined[:, 0::2] = sin_feats
        combined[:, 1::2] = cos_feats
        features_list.append(combined)
    Sfull = np.concatenate(features_list, axis=1)
    perm = rng.permutation(Sfull.shape[1])
    Sperm = Sfull[:, perm][:, :L]
    Sperm = Sperm / math.sqrt(L)
    return Sperm


# ------------------------- Regression & prediction -------------------------

def fit_ridge(S_train: np.ndarray, y_train: np.ndarray, z: float) -> np.ndarray:
    clf = Ridge(alpha=float(z), fit_intercept=False, solver="auto")
    clf.fit(S_train, y_train)
    return clf.coef_


def rolling_window_predictions(S: np.ndarray, returns: np.ndarray, T_window: int, z: float) -> np.ndarray:
    N, L = S.shape
    preds = np.full(N, np.nan, dtype=float)
    for t in range(T_window, N):
        S_train = S[t - T_window:t, :]
        y_train = returns[t - T_window:t]
        if np.isnan(S_train).any() or np.isnan(y_train).any():
            continue
        beta = fit_ridge(S_train, y_train, z)
        preds[t] = float(S[t, :] @ beta)
    return preds


def fast_rolling_predictions(S: np.ndarray, returns: np.ndarray, T_window: int, z: float) -> np.ndarray:
    """Faster rolling predictions using incremental updates of S'S and S'y.

    Important: S and returns must be aligned and free of NaNs on the rows used for training.
    The calling code will drop NaNs before generating S.
    """
    N, L = S.shape
    preds = np.full(N, np.nan, dtype=float)
    lam = float(z)
    if N < T_window or T_window <= 0:
        return preds
    # initialize
    STS = S[:T_window].T @ S[:T_window]
    STy = S[:T_window].T @ returns[:T_window]
    I = lam * np.eye(L)
    for t in range(T_window, N):
        # solve for beta
        try:
            beta = np.linalg.solve(STS + I, STy)
        except np.linalg.LinAlgError:
            # numerical issue — fall back to pseudo-inverse
            beta = np.linalg.pinv(STS + I) @ STy
        preds[t] = float(S[t] @ beta)
        if t < N - 1:
            # slide window forward by removing oldest and adding current
            old = S[t - T_window]
            new = S[t]
            STS = STS - np.outer(old, old) + np.outer(new, new)
            STy = STy - old * returns[t - T_window] + new * returns[t]
    return preds


# ------------------------- Metrics & experiment wrapper -------------------------

def compute_metrics(returns: np.ndarray, preds: np.ndarray) -> dict:
    mask = ~np.isnan(preds)
    if mask.sum() == 0:
        return {"r2": np.nan, "ER": np.nan, "Vol": np.nan, "SR": np.nan}
    y = returns[mask]
    p = preds[mask]
    Rhat = p * y
    sse = np.sum((y - p) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    ER = float(np.mean(Rhat))
    Vol = float(np.mean((Rhat - ER) ** 2))
    SR = float(ER / math.sqrt(Vol)) if Vol > 0 else np.nan
    return {"r2": r2, "ER": ER, "Vol": Vol, "SR": SR}


def run_experiment(returns: pd.Series, X_raw: pd.DataFrame, L: int, T_window: int, z: float, gammas: Sequence[float], seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Align data, generate features, run rolling predictions, and compute metrics."""
    df = pd.concat([returns.rename("return"), X_raw], axis=1).dropna()
    if df.shape[0] == 0:
        raise ValueError("No overlapping non-NaN rows between returns and signals after alignment.")
    ret = df["return"].to_numpy(dtype=float)
    X = df.drop(columns=["return"]).to_numpy(dtype=float)
    S = generate_random_fourier_features(X, L=L, gammas=gammas, seed=seed)
    preds = fast_rolling_predictions(S, ret, T_window, z)
    metrics = compute_metrics(ret, preds)
    return preds, metrics


# ------------------------- Sweeps and plotting -------------------------

def sweep_experiment(returns: pd.Series, X_raw: pd.DataFrame, L_list: Sequence[int], T_list: Sequence[int], z_list: Sequence[float], gammas: Sequence[float], seeds: Sequence[int]) -> pd.DataFrame:
    """Run multi-parameter sweeps over L, T, z, seeds.  Align data once (drop NaNs) then generate features per (L,seed).

    Returns: DataFrame with columns L, T, z, seed, r2, ER, Vol, SR
    """
    # align once and drop NaNs so all S and ret arrays are consistent in length
    df = pd.concat([returns.rename("return"), X_raw], axis=1).dropna()
    if df.shape[0] == 0:
        raise ValueError("No overlapping non-NaN rows between returns and signals after alignment.")
    ret_array = df["return"].to_numpy(dtype=float)
    X_array = df.drop(columns=["return"]).to_numpy(dtype=float)

    records = []
    for L in L_list:
        S_cache = {}
        for seed in seeds:
            S_cache[seed] = generate_random_fourier_features(X_array, L=L, gammas=gammas, seed=seed)
        for T_window in T_list:
            for z in z_list:
                for seed in seeds:
                    S = S_cache[seed]
                    preds = fast_rolling_predictions(S, ret_array, T_window, z)
                    metrics = compute_metrics(ret_array, preds)
                    record = dict(L=L, T=T_window, z=z, seed=seed)
                    record.update(metrics)
                    records.append(record)
    return pd.DataFrame.from_records(records)


def plot_voc_curve(df_results: pd.DataFrame, metric: str = "SR"):
    if metric not in df_results.columns:
        raise ValueError(f"Metric '{metric}' not in results DataFrame")
    grouped = df_results.groupby(["T", "z", "L"])[metric].mean().reset_index()
    for (T_val, z_val), g in grouped.groupby(["T", "z"]):
        plt.plot(g["L"], g[metric], marker="o", label=f"T={T_val}, z={z_val}")
    plt.xlabel("Number of Random Features L")
    plt.ylabel(metric)
    plt.title(f"Value of Complexity ({metric})")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------- Demo & CLI -------------------------

def _make_synthetic_demo(n=400, d=3, noise_std=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.sin(X[:, 0] * 1.5) + 0.5 * (X[:, 1] ** 2) - 0.3 * X[:, 2] + rng.normal(scale=noise_std, size=n)
    idx = pd.date_range("2000-01-01", periods=n, freq="ME")
    return pd.Series(y, index=idx), pd.DataFrame(X, index=idx, columns=[f"x{i}" for i in range(d)])


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV of returns with a column named 'return' (optional)")
    parser.add_argument("--date-col", help="Date column in CSV (optional)")
    parser.add_argument("--T", type=int, nargs="*", default=[120], help="rolling window T (can give multiple)")
    parser.add_argument("--L", type=int, nargs="*", default=[500], help="number of random features L (can give multiple)")
    parser.add_argument("--z", type=float, nargs="*", default=[1.0], help="ridge penalty z (can give multiple)")
    parser.add_argument("--seed", type=int, nargs="*", default=[0], help="random seed(s)")
    parser.add_argument("--demo", action="store_true", help="run synthetic demo instead of CSV")
    parser.add_argument("--sweep", action="store_true", help="run multi-parameter sweep")
    parser.add_argument("--plot", action="store_true", help="plot VoC curve for sweep results")
    parser.add_argument("--metric", type=str, default="SR", help="metric for plotting")
    args = parser.parse_args()

    gammas = [0.1, 0.5, 1, 2, 4, 8, 16]
    if args.demo or args.csv is None:
        returns, X = _make_synthetic_demo(n=400, d=3, seed=args.seed[0])
    else:
        returns = load_returns(args.csv, date_col=args.date_col, return_col="return")
        X = pd.DataFrame({"lag1": returns.shift(1), "ma12": returns.rolling(window=12, min_periods=1).mean()})

    # standardize and then rely on run_experiment / sweep_experiment to align and drop NaNs
    ret_std = standardize_returns(returns, window=12)

    if args.sweep:
        df_results = sweep_experiment(ret_std, X, L_list=args.L, T_list=args.T, z_list=args.z, gammas=gammas, seeds=args.seed)
        print(df_results)
        if args.plot:
            plot_voc_curve(df_results, metric=args.metric)
    else:
        preds, metrics = run_experiment(ret_std, X, L=args.L[0], T_window=args.T[0], z=args.z[0], gammas=gammas, seed=args.seed[0])
        print("Experiment finished. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


# ------------------------- Unit tests (pytest) -------------------------

import pytest


class TestVoc:
    def test_generate_rff_shape(self):
        X = np.zeros((50, 4))
        S = generate_random_fourier_features(X, L=100, gammas=[1.0, 2.0], seed=1)
        assert S.shape == (50, 100)

    def test_fit_ridge_perfect(self):
        rng = np.random.default_rng(2)
        S = rng.normal(size=(30, 5))
        beta = rng.normal(size=5)
        y = S @ beta
        beta_hat = fit_ridge(S, y, z=1e-6)
        assert np.allclose(beta_hat, beta, atol=1e-5)

    def test_rolling_preds_nan_for_short(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 2))
        S = generate_random_fourier_features(X, L=10, gammas=[1.0], seed=3)
        returns = rng.normal(size=20)
        preds = rolling_window_predictions(S, returns, T_window=25, z=1.0)
        assert np.isnan(preds).all()

    def test_metrics_known(self):
        rng = np.random.default_rng(4)
        y = rng.normal(size=100)
        preds = y.copy()
        metrics = compute_metrics(y, preds)
        assert pytest.approx(metrics["r2"], rel=1e-6) == 1.0


if __name__ == "__main__":
    main_cli()
