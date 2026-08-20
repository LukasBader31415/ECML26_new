# single_matrix_analysis.py
# Ziel:
# - Analyse einer einzelnen Distanz-/Dissimilarity-Matrix
# - Output so strukturieren wie bei der MGLVQ-Pipeline:
#     results.pkl + summary.csv/pkl + optional point_table.pkl
# - Zusätzlich objekt-level Features (purity_i, margin_i, knn_mean_i, etc.),
#   um später "structural goodness" mit "correct / wrong predictions"
#   verknüpfen zu können.

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd


# =========================
# tie-safe kNN purity (ECML26 Neuerung 1)
# =========================
def _knn_purity_tiesafe(D, y, k):
    """Fraction of same-class neighbours among the k nearest, self excluded.
    Ties at the k-th distance are shared proportionally so the effective
    neighbourhood size stays exactly k (fixes the np.argsort index-order bias
    on categorical views with ~90-100% ties). mode == 'prop'."""
    D = np.asarray(D, dtype=float); y = np.asarray(y)
    m = D.shape[0]; pur = np.empty(m)
    for i in range(m):
        d = D[i].copy(); d[i] = np.inf
        thr = np.partition(d, k - 1)[k - 1]
        below = d < thr; at = d == thr
        n_below = int(below.sum()); rem = k - n_below
        w = np.zeros(m); w[below] = 1.0
        n_at = int(at.sum())
        if n_at and rem > 0:
            w[at] = rem / n_at
        pur[i] = float((w * (y == y[i])).sum()) / k
    return pur


def _knn_purity_weights(D, k):
    """Row-normalised tie-safe neighbour weights (m x m, rows sum to k), so a
    permutation baseline can reuse fixed weights with relabelled y."""
    D = np.asarray(D, dtype=float); m = D.shape[0]
    W = np.zeros((m, m))
    for i in range(m):
        d = D[i].copy(); d[i] = np.inf
        thr = np.partition(d, k - 1)[k - 1]
        below = d < thr; at = d == thr
        rem = k - int(below.sum()); n_at = int(at.sum())
        W[i, below] = 1.0
        if n_at and rem > 0:
            W[i, at] = rem / n_at
    return W


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union


# =========================
# 0) Data containers
# =========================

@dataclass
class MatrixBasicChecks:
    n: int
    symmetric_max_abs_diff: float
    diagonal_min: float
    diagonal_max: float
    value_min: float
    value_max: float
    value_mean: float
    value_std: float


@dataclass
class MatrixDistributionStats:
    quantiles: Dict[float, float]
    mean: float
    std: float
    cv: float
    skew_approx: Optional[float] = None


@dataclass
class MatrixGeometryStats:
    row_mean_stats: pd.Series
    knn_mean_stats: pd.Series
    knn_global_mean: float
    knn_global_std: float
    intrinsic_dim_2nn: float


@dataclass
class MatrixDiscriminativeStats:
    class_counts: Dict[int, int]
    intra_mean: float
    intra_std: float
    inter_mean: float
    inter_std: float
    separation_ratio: float
    margin_stats: pd.Series
    margin_fraction_negative: float
    knn_purity_stats: pd.Series
    knn_purity_mean: float


# =========================
# 1) Utilities
# =========================

def _as_numpy_square(D_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if D_df.shape[0] != D_df.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {D_df.shape}")

    if not D_df.index.equals(D_df.columns):
        if set(D_df.index) != set(D_df.columns):
            raise ValueError("Index and columns must contain the same IDs.")
        D_df = D_df.loc[D_df.index, D_df.index]

    labels = D_df.index.astype(str).tolist()
    D = D_df.to_numpy()
    return D, labels


def _validate_y(y: pd.Series | np.ndarray, n: int) -> np.ndarray:
    y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if y_arr.shape[0] != n:
        raise ValueError(
            f"y must have length n={n}, got {y_arr.shape[0]}. "
            "Empfehlung: y als pd.Series mit Index=IDs übergeben und Matrix vorher alignen."
        )
    return y_arr.astype(int)


def _safe_quantiles(
    x: np.ndarray,
    qs=(0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0),
) -> Dict[float, float]:
    qv = np.quantile(x, qs)
    return {float(q): float(v) for q, v in zip(qs, qv)}


# =========================
# 2) Step-by-step functions
# =========================

def step1_basic_checks(
    D_df: pd.DataFrame,
    tol_sym: float = 1e-8,
) -> Tuple[np.ndarray, List[str], MatrixBasicChecks]:
    D, ids = _as_numpy_square(D_df)
    sym = float(np.max(np.abs(D - D.T)))
    diag = np.diag(D)

    checks = MatrixBasicChecks(
        n=D.shape[0],
        symmetric_max_abs_diff=sym,
        diagonal_min=float(diag.min()),
        diagonal_max=float(diag.max()),
        value_min=float(D.min()),
        value_max=float(D.max()),
        value_mean=float(D.mean()),
        value_std=float(D.std()),
    )

    if sym > tol_sym:
        print(f"[WARN] Matrix not symmetric within tol={tol_sym}. max|D-D.T|={sym:.3e}")
    if not np.isfinite(D).all():
        raise ValueError("Matrix contains NaN or inf.")

    return D, ids, checks


def step2_distance_distribution(
    D: np.ndarray,
    quantiles=(0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0),
) -> MatrixDistributionStats:
    iu = np.triu_indices(D.shape[0], k=1)
    vals = D[iu]

    mean = float(vals.mean())
    std = float(vals.std())
    cv = float(std / mean) if mean > 0 else float("nan")

    return MatrixDistributionStats(
        quantiles=_safe_quantiles(vals, qs=quantiles),
        mean=mean,
        std=std,
        cv=cv,
    )


def step3_geometry_local_global(
    D: np.ndarray,
    k: int = 10,
    intrinsic_dim_eps: float = 1e-12,
) -> MatrixGeometryStats:
    n = D.shape[0]
    if k >= n:
        raise ValueError(f"k must be < n, got k={k}, n={n}")

    row_mean = D.mean(axis=1)
    row_mean_stats = pd.Series(row_mean).describe()

    knn_d = np.partition(D, kth=k, axis=1)[:, 1:k + 1]
    knn_mean_per_point = knn_d.mean(axis=1)
    knn_mean_stats = pd.Series(knn_mean_per_point).describe()

    knn_global_mean = float(knn_d.mean())
    knn_global_std = float(knn_d.std())

    nn2 = np.partition(D, kth=2, axis=1)[:, 1:3]
    r1 = nn2[:, 0]
    r2 = nn2[:, 1]
    mask = (r1 > intrinsic_dim_eps) & (r2 > r1)
    mu = r2[mask] / r1[mask]

    if mask.sum() < max(50, int(0.05 * n)):
        print(f"[WARN] Few valid points for 2NN intrinsic dimension: {mask.sum()} / {n}")

    id_hat = float(1.0 / np.mean(np.log(mu))) if mu.size > 0 else float("nan")

    return MatrixGeometryStats(
        row_mean_stats=row_mean_stats,
        knn_mean_stats=knn_mean_stats,
        knn_global_mean=knn_global_mean,
        knn_global_std=knn_global_std,
        intrinsic_dim_2nn=id_hat,
    )


def step4_discriminative_power(
    D: np.ndarray,
    y: pd.Series | np.ndarray,
    k: int = 10,
    eps: float = 1e-12,
) -> Tuple[MatrixDiscriminativeStats, Dict[str, np.ndarray]]:
    """
    Returns:
      - aggregate discriminative stats
      - point-level arrays needed for linking to MGLVQ:
          d_plus, d_minus, mu, purity, knn_mean
    """
    n = D.shape[0]
    if k >= n:
        raise ValueError(f"k must be < n, got k={k}, n={n}")

    y_arr = _validate_y(y, n)

    unique, counts = np.unique(y_arr, return_counts=True)
    class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    iu = np.triu_indices(n, k=1)
    du = D[iu]
    same = (y_arr[iu[0]] == y_arr[iu[1]])

    intra = du[same]
    inter = du[~same]

    intra_mean = float(intra.mean()) if intra.size else float("nan")
    inter_mean = float(inter.mean()) if inter.size else float("nan")
    intra_std = float(intra.std()) if intra.size else float("nan")
    inter_std = float(inter.std()) if inter.size else float("nan")
    sep_ratio = float(inter_mean / intra_mean) if intra_mean > 0 else float("nan")

    idx_by_class = {c: np.where(y_arr == c)[0] for c in unique}

    d_plus = np.empty(n, dtype=float)
    d_minus = np.empty(n, dtype=float)

    for t in range(n):
        c = y_arr[t]
        same_idx = idx_by_class[c]
        other_idx = np.where(y_arr != c)[0]

        ds = D[t, same_idx].copy()
        ds[same_idx == t] = np.inf
        d_plus[t] = float(np.min(ds))

        do = D[t, other_idx]
        d_minus[t] = float(np.min(do))

    mu = (d_plus - d_minus) / (d_plus + d_minus + eps)

    purity = _knn_purity_tiesafe(D, y_arr, k)

    knn_d = np.partition(D, kth=k, axis=1)[:, 1:k + 1]
    knn_mean = knn_d.mean(axis=1)

    margin_stats = pd.Series(mu).describe()
    frac_neg = float(np.mean(mu < 0))

    purity_stats = pd.Series(purity).describe()
    purity_mean = float(purity.mean())

    agg = MatrixDiscriminativeStats(
        class_counts=class_counts,
        intra_mean=intra_mean,
        intra_std=intra_std,
        inter_mean=inter_mean,
        inter_std=inter_std,
        separation_ratio=sep_ratio,
        margin_stats=margin_stats,
        margin_fraction_negative=frac_neg,
        knn_purity_stats=purity_stats,
        knn_purity_mean=purity_mean,
    )

    point = {
        "d_plus": d_plus,
        "d_minus": d_minus,
        "mu": mu,
        "knn_purity": purity,
        "knn_mean": knn_mean,
    }
    return agg, point


# =========================
# 2b) Zusatzdiagnostik
# =========================

def zero_distance_stats(D: np.ndarray) -> dict:
    n = D.shape[0]
    du = D[np.triu_indices(n, k=1)]
    zero_rate_global = float(np.mean(du == 0.0))

    zero_per_row = (D == 0.0).sum(axis=1) - 1
    return {
        "zero_rate_global": zero_rate_global,
        "zero_per_row_describe": pd.Series(zero_per_row).describe(),
        "fraction_rows_with_any_zero_neighbor": float(np.mean(zero_per_row > 0)),
        "fraction_rows_with_10plus_zero_neighbors": float(np.mean(zero_per_row >= 10)),
    }


def unique_distance_values(D: np.ndarray, max_show: int = 30) -> dict:
    n = D.shape[0]
    du = D[np.triu_indices(n, k=1)]
    uniq = np.unique(du)

    out = {"n_unique": int(len(uniq))}
    if len(uniq) <= max_show:
        out["values"] = uniq.tolist()
    else:
        out["values_head"] = uniq[:max_show].tolist()
        out["values_tail"] = uniq[-max_show:].tolist()

    return out


def knn_purity_with_baseline(
    D: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    n_perm: int = 50,
    seed: int = 0,
) -> dict:
    n = D.shape[0]
    if k >= n:
        raise ValueError(f"k must be < n, got k={k}, n={n}")

    y = np.asarray(y).astype(int)
    W = _knn_purity_weights(D, k)
    purity = (W * (y[:, None] == y[None, :])).sum(axis=1) / k
    purity_mean = float(purity.mean())

    rng = np.random.default_rng(seed)
    baseline = np.empty(n_perm, dtype=float)
    for t in range(n_perm):
        yp = rng.permutation(y)
        baseline[t] = float(((W * (yp[:, None] == yp[None, :])).sum(axis=1) / k).mean())

    return {
        "purity_mean": purity_mean,
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std()),
        "z_score": float((purity_mean - baseline.mean()) / (baseline.std() + 1e-12)),
        "purity_describe": pd.Series(purity).describe(),
    }


def margin_stats_no_zero_ties(
    D: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    y = np.asarray(y).astype(int)
    n = D.shape[0]
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}

    d_plus = np.empty(n, dtype=float)
    d_minus = np.empty(n, dtype=float)

    for t in range(n):
        c = y[t]
        same_idx = idx_by_class[c]
        other_idx = np.where(y != c)[0]

        ds = D[t, same_idx].copy()
        ds[same_idx == t] = np.inf
        ds_pos = ds[ds > 0]
        d_plus[t] = float(ds_pos.min()) if ds_pos.size else float(ds.min())

        do = D[t, other_idx]
        do_pos = do[do > 0]
        d_minus[t] = float(do_pos.min()) if do_pos.size else float(do.min())

    mu = (d_plus - d_minus) / (d_plus + d_minus + eps)
    return {
        "mu_describe": pd.Series(mu).describe(),
        "fraction_mu_negative": float(np.mean(mu < 0)),
        "fraction_mu_zero": float(np.mean(mu == 0)),
    }


# =========================
# 3) Orchestrator (returns + saving)
# =========================

def run_single_matrix_analysis(
    D_df: pd.DataFrame,
    y: Optional[pd.Series | np.ndarray] = None,
    k: int = 10,
    tol_sym: float = 1e-8,
    n_perm: int = 50,
    seed: int = 0,
    *,
    matrix_name: str = "matrix",
    out_dir: Optional[Union[str, Path]] = None,
    save_point_table: bool = True,
) -> Dict[str, object]:
    """
    If out_dir is provided:
      out_dir/<matrix_name>/
        results.pkl
        summary.pkl
        summary.csv
        point_table.pkl
        used_ids.pkl
    """
    D, ids, basic = step1_basic_checks(D_df, tol_sym=tol_sym)
    dist = step2_distance_distribution(D)
    geom = step3_geometry_local_global(D, k=k)

    results: Dict[str, object] = {
        "matrix_name": matrix_name,
        "used_ids": ids,
        "basic": basic,
        "distribution": dist,
        "geometry": geom,
        "zero_stats": zero_distance_stats(D),
        "unique_values": unique_distance_values(D),
    }

    point_table = None
    if y is not None:
        disc, point = step4_discriminative_power(D, y, k=k)
        results["discriminative"] = disc
        results["knn_baseline"] = knn_purity_with_baseline(D, np.asarray(y), k=k, n_perm=n_perm, seed=seed)
        results["robust_margin"] = margin_stats_no_zero_ties(D, np.asarray(y))

        y_arr = _validate_y(y, len(ids))
        point_table = pd.DataFrame(
            {
                "id": ids,
                "y": y_arr.astype(int),
                "mu": point["mu"],
                "d_plus": point["d_plus"],
                "d_minus": point["d_minus"],
                "knn_purity": point["knn_purity"],
                "knn_mean": point["knn_mean"],
            }
        ).set_index("id")

    if out_dir is not None:
        out_dir = Path(out_dir) / matrix_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "results.pkl").open("wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        summary_df = results_to_summary_df(matrix_name, results)
        with (out_dir / "summary.pkl").open("wb") as f:
            pickle.dump(summary_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        summary_df.to_csv(out_dir / "summary.csv", index=True)

        with (out_dir / "used_ids.pkl").open("wb") as f:
            pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)

        if save_point_table and point_table is not None:
            with (out_dir / "point_table.pkl").open("wb") as f:
                pickle.dump(point_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    if point_table is not None:
        results["point_table"] = point_table

    return results


# =========================
# 4) Pretty printer
# =========================

def print_single_matrix_report(results: Dict[str, object]) -> None:
    basic: MatrixBasicChecks = results["basic"]
    dist: MatrixDistributionStats = results["distribution"]
    geom: MatrixGeometryStats = results["geometry"]

    print("=== BASIC CHECKS ===")
    print(basic)

    print("\n=== DISTANCE DISTRIBUTION (upper triangle) ===")
    print("mean/std/cv:", dist.mean, dist.std, dist.cv)
    print("quantiles:", dist.quantiles)

    print("\n=== GEOMETRY ===")
    print("row_mean describe:\n", geom.row_mean_stats)
    print("\nknn_mean_per_point describe:\n", geom.knn_mean_stats)
    print("\nknn global mean/std:", geom.knn_global_mean, geom.knn_global_std)
    print("\n2NN intrinsic dimension:", geom.intrinsic_dim_2nn)

    print("\n=== ZERO DISTANCE ANALYSIS ===")
    print(results.get("zero_stats"))

    print("\n=== UNIQUE DISTANCE VALUES ===")
    print(results.get("unique_values"))

    if "knn_baseline" in results:
        print("\n=== kNN PURITY WITH RANDOM BASELINE ===")
        print(results["knn_baseline"])

    if "robust_margin" in results:
        print("\n=== ROBUST MARGIN (no-zero ties) ===")
        print(results["robust_margin"])

    if "discriminative" in results:
        disc: MatrixDiscriminativeStats = results["discriminative"]
        print("\n=== DISCRIMINATIVE (with labels) ===")
        print("class_counts:", disc.class_counts)
        print("intra mean/std:", disc.intra_mean, disc.intra_std)
        print("inter mean/std:", disc.inter_mean, disc.inter_std)
        print("separation ratio:", disc.separation_ratio)
        print("\nmargin describe:\n", disc.margin_stats)
        print("fraction mu<0:", disc.margin_fraction_negative)
        print("\nknn purity describe:\n", disc.knn_purity_stats)
        print("knn purity mean:", disc.knn_purity_mean)


# =========================
# 5) Flatten to table row
# =========================

def flatten_results_to_row(matrix_name: str, results: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, Any] = {"matrix": matrix_name}

    basic = results.get("basic")
    if basic is not None:
        row.update(
            {
                "n": basic.n,
                "sym_max_abs_diff": basic.symmetric_max_abs_diff,
                "diag_min": basic.diagonal_min,
                "diag_max": basic.diagonal_max,
                "D_min": basic.value_min,
                "D_max": basic.value_max,
                "D_mean_all": basic.value_mean,
                "D_std_all": basic.value_std,
            }
        )

    dist = results.get("distribution")
    if dist is not None:
        q = dist.quantiles or {}
        row.update(
            {
                "dist_mean_ut": dist.mean,
                "dist_std_ut": dist.std,
                "dist_cv_ut": dist.cv,
                "q0": q.get(0.0, np.nan),
                "q01": q.get(0.01, np.nan),
                "q05": q.get(0.05, np.nan),
                "q50": q.get(0.5, np.nan),
                "q95": q.get(0.95, np.nan),
                "q99": q.get(0.99, np.nan),
                "q100": q.get(1.0, np.nan),
            }
        )

    geom = results.get("geometry")
    if geom is not None:
        rm = geom.row_mean_stats
        km = geom.knn_mean_stats
        row.update(
            {
                "row_mean_mean": float(rm.get("mean", np.nan)),
                "row_mean_std": float(rm.get("std", np.nan)),
                "knn_mean_mean": float(km.get("mean", np.nan)),
                "knn_mean_std": float(km.get("std", np.nan)),
                "knn_global_mean": geom.knn_global_mean,
                "knn_global_std": geom.knn_global_std,
                "intrinsic_dim_2nn": geom.intrinsic_dim_2nn,
            }
        )

    zs = results.get("zero_stats", {})
    if isinstance(zs, dict):
        row.update(
            {
                "zero_rate_global": zs.get("zero_rate_global", np.nan),
                "zero_rows_any": zs.get("fraction_rows_with_any_zero_neighbor", np.nan),
                "zero_rows_10plus": zs.get("fraction_rows_with_10plus_zero_neighbors", np.nan),
            }
        )
        zdesc = zs.get("zero_per_row_describe")
        if isinstance(zdesc, pd.Series):
            row.update(
                {
                    "zero_per_row_mean": float(zdesc.get("mean", np.nan)),
                    "zero_per_row_std": float(zdesc.get("std", np.nan)),
                    "zero_per_row_max": float(zdesc.get("max", np.nan)),
                }
            )

    uv = results.get("unique_values", {})
    if isinstance(uv, dict):
        row["n_unique_dist_values"] = uv.get("n_unique", np.nan)

    disc = results.get("discriminative")
    if disc is not None:
        row.update(
            {
                "intra_mean": disc.intra_mean,
                "inter_mean": disc.inter_mean,
                "sep_ratio_inter_intra": disc.separation_ratio,
                "margin_mu_mean": float(disc.margin_stats.get("mean", np.nan)),
                "margin_frac_neg": disc.margin_fraction_negative,
                "knn_purity_mean": disc.knn_purity_mean,
            }
        )
        if isinstance(disc.class_counts, dict):
            row["n_class0"] = disc.class_counts.get(0, np.nan)
            row["n_class1"] = disc.class_counts.get(1, np.nan)

    kb = results.get("knn_baseline", {})
    if isinstance(kb, dict):
        row.update(
            {
                "knn_purity_baseline_mean": kb.get("baseline_mean", np.nan),
                "knn_purity_baseline_std": kb.get("baseline_std", np.nan),
                "knn_purity_z": kb.get("z_score", np.nan),
            }
        )

    rm2 = results.get("robust_margin", {})
    if isinstance(rm2, dict):
        mu_desc = rm2.get("mu_describe")
        if isinstance(mu_desc, pd.Series):
            row["robust_mu_mean"] = float(mu_desc.get("mean", np.nan))
            row["robust_mu_std"] = float(mu_desc.get("std", np.nan))
        row["robust_mu_frac_neg"] = rm2.get("fraction_mu_negative", np.nan)
        row["robust_mu_frac_zero"] = rm2.get("fraction_mu_zero", np.nan)

    return row


def results_to_summary_df(matrix_name: str, results: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([flatten_results_to_row(matrix_name, results)]).set_index("matrix")