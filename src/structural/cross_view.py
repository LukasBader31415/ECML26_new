from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# =========================
# 1) Matrix Correlation
# =========================

def _validate_square_and_ids(matrices: Dict[str, pd.DataFrame]) -> None:
    if not matrices:
        raise ValueError("matrices ist leer.")

    for name, D in matrices.items():
        if not isinstance(D, pd.DataFrame):
            raise TypeError(f"{name}: erwartet pd.DataFrame, bekam {type(D)}")
        if D.shape[0] != D.shape[1]:
            raise ValueError(f"{name}: Matrix ist nicht quadratisch: {D.shape}")
        if set(D.index) != set(D.columns):
            raise ValueError(f"{name}: Index und Columns haben nicht dieselben IDs.")


def _subset_to_ids(
    matrices: Dict[str, pd.DataFrame],
    ids: Optional[pd.Index] = None,
) -> Dict[str, pd.DataFrame]:
    _validate_square_and_ids(matrices)

    if ids is None:
        names = list(matrices.keys())
        ref_idx = matrices[names[0]].index
        ref_col = matrices[names[0]].columns

        for name, D in matrices.items():
            if not D.index.equals(ref_idx) or not D.columns.equals(ref_col):
                raise ValueError(
                    f"{name}: Matrix ist nicht auf denselben Canvas aligned wie die anderen. "
                    "Bitte vorher zentral alignen."
                )
        return matrices

    ids = pd.Index(ids)
    aligned = {}
    for name, D in matrices.items():
        missing = set(ids) - set(D.index)
        if missing:
            ex = ", ".join(list(sorted(missing))[:5])
            more = " ..." if len(missing) > 5 else ""
            raise ValueError(f"{name}: enthält nicht alle vorgegebenen ids: {ex}{more}")
        aligned[name] = D.loc[ids, ids]

    return aligned


def _upper_triangle_vector(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu].astype(float, copy=False)


def pairwise_matrix_correlation(
    matrices: Dict[str, pd.DataFrame],
    method: str = "pearson",
    ids: Optional[pd.Index] = None,
) -> pd.DataFrame:
    aligned = _subset_to_ids(matrices, ids=ids)
    names = list(aligned.keys())

    vecs = {}
    for name in names:
        D = aligned[name].to_numpy()
        vecs[name] = _upper_triangle_vector(D)

    C = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                C.loc[a, b] = 1.0
                continue
            if j < i:
                C.loc[a, b] = C.loc[b, a]
                continue

            va, vb = vecs[a], vecs[b]

            if method == "pearson":
                C.loc[a, b] = float(np.corrcoef(va, vb)[0, 1])
            elif method == "spearman":
                ra = pd.Series(va).rank(method="average").to_numpy()
                rb = pd.Series(vb).rank(method="average").to_numpy()
                C.loc[a, b] = float(np.corrcoef(ra, rb)[0, 1])
            else:
                raise ValueError("method muss 'pearson' oder 'spearman' sein.")

    return C


def correlation_summary(corr_df: pd.DataFrame) -> pd.DataFrame:
    names = list(corr_df.index)
    out = []

    for a in names:
        others = [b for b in names if b != a]
        vals = corr_df.loc[a, others].to_numpy(dtype=float)
        out.append({
            "matrix": a,
            "mean_abs_corr_to_others": float(np.mean(np.abs(vals))) if len(vals) else np.nan,
            "max_abs_corr_to_others": float(np.max(np.abs(vals))) if len(vals) else np.nan,
        })

    return (
        pd.DataFrame(out)
        .set_index("matrix")
        .sort_values("mean_abs_corr_to_others")
    )


# =========================
# 2) kNN Overlap
# =========================

def _validate_canvas_matrices(
    matrices: Dict[str, pd.DataFrame],
    ids: Optional[pd.Index] = None,
) -> pd.Index:
    if not matrices:
        raise ValueError("matrices ist leer.")

    names = list(matrices.keys())
    first = matrices[names[0]]

    if first.shape[0] != first.shape[1]:
        raise ValueError(f"{names[0]}: Matrix ist nicht quadratisch: {first.shape}")
    if not first.index.equals(first.columns):
        raise ValueError(f"{names[0]}: Index und Columns haben nicht dieselben IDs in gleicher Reihenfolge.")

    ref_ids = pd.Index(first.index)

    if ids is not None:
        ids = pd.Index(ids)
        if not ref_ids.equals(ids):
            raise ValueError("Die erste Matrix passt nicht zu den vorgegebenen ids.")

    for name in names[1:]:
        D = matrices[name]
        if D.shape[0] != D.shape[1]:
            raise ValueError(f"{name}: Matrix ist nicht quadratisch: {D.shape}")
        if not D.index.equals(D.columns):
            raise ValueError(f"{name}: Index und Columns haben nicht dieselben IDs in gleicher Reihenfolge.")
        if not D.index.equals(ref_ids):
            raise ValueError(
                f"{name}: Matrix ist nicht auf demselben Canvas aligned. "
                "Alle Matrizen müssen exakt dieselben IDs in derselben Reihenfolge haben."
            )

    return ref_ids


def knn_indices_from_distance_matrix(D: np.ndarray, k: int = 10) -> np.ndarray:
    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D muss quadratisch sein, got shape={D.shape}")
    if k <= 0:
        raise ValueError("k muss > 0 sein.")
    if k >= D.shape[0]:
        raise ValueError(f"k={k} ist zu groß für n={D.shape[0]}; benötigt k < n.")

    return np.argsort(D, axis=1)[:, 1:k + 1]


def _jaccard_rowwise(neigh_a: np.ndarray, neigh_b: np.ndarray) -> np.ndarray:
    if neigh_a.shape != neigh_b.shape:
        raise ValueError(f"Shape mismatch: {neigh_a.shape} vs {neigh_b.shape}")

    n, _ = neigh_a.shape
    out = np.empty(n, dtype=float)

    for i in range(n):
        sa = set(neigh_a[i].tolist())
        sb = set(neigh_b[i].tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        out[i] = inter / union if union > 0 else 0.0

    return out


def pairwise_knn_overlap(
    matrices: Dict[str, pd.DataFrame],
    k: int = 10,
    ids: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.Series], pd.Index]:
    used_ids = _validate_canvas_matrices(matrices, ids=ids)
    names = list(matrices.keys())

    knn_map: Dict[str, np.ndarray] = {}
    for name in names:
        D = matrices[name].to_numpy()
        knn_map[name] = knn_indices_from_distance_matrix(D, k=k)

    overlap_mean = pd.DataFrame(index=names, columns=names, dtype=float)
    overlap_dist: Dict[Tuple[str, str], pd.Series] = {}

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                overlap_mean.loc[a, b] = 1.0
                continue
            if j < i:
                overlap_mean.loc[a, b] = overlap_mean.loc[b, a]
                continue

            ov = _jaccard_rowwise(knn_map[a], knn_map[b])
            overlap_mean.loc[a, b] = float(ov.mean())
            overlap_dist[(a, b)] = pd.Series(
                ov,
                index=used_ids,
                name=f"overlap_{a}_{b}",
            )

    return overlap_mean, overlap_dist, used_ids


def overlap_summary(overlap_mean_df: pd.DataFrame) -> pd.DataFrame:
    names = list(overlap_mean_df.index)
    out = []

    for a in names:
        others = [b for b in names if b != a]
        vals = overlap_mean_df.loc[a, others].to_numpy(dtype=float)

        out.append({
            "matrix": a,
            "mean_knn_overlap_to_others": float(np.mean(vals)) if len(vals) else np.nan,
            "min_knn_overlap_to_others": float(np.min(vals)) if len(vals) else np.nan,
            "max_knn_overlap_to_others": float(np.max(vals)) if len(vals) else np.nan,
        })

    return pd.DataFrame(out).set_index("matrix").sort_values("mean_knn_overlap_to_others")


# =========================
# 3) Agreement / Disagreement
# =========================

def _coerce_and_slice_matrices(
    matrices: Dict[str, pd.DataFrame],
    ids: Optional[pd.Index] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.Index]:
    _validate_square_and_ids(matrices)

    names = list(matrices.keys())
    first = matrices[names[0]]

    if ids is None:
        ref_idx = first.index
        ref_cols = first.columns

        for name, D in matrices.items():
            if not D.index.equals(ref_idx) or not D.columns.equals(ref_cols):
                raise ValueError(
                    f"{name}: Matrix ist nicht auf derselben Leinwand wie die anderen Matrizen. "
                    "Bitte vorher zentral alignen oder ids explizit übergeben."
                )

        used_ids = ref_idx
        aligned = matrices
        return aligned, used_ids

    used_ids = pd.Index(ids)

    aligned: Dict[str, pd.DataFrame] = {}
    for name, D in matrices.items():
        missing = set(used_ids) - set(D.index)
        if missing:
            ex = list(sorted(missing))[:5]
            more = " ..." if len(missing) > 5 else ""
            raise ValueError(f"{name}: enthält nicht alle ids. Missing: {ex}{more}")
        aligned[name] = D.loc[used_ids, used_ids]

    return aligned, used_ids


def _upper_triangle(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def thresholds_by_quantile(
    matrices: Dict[str, pd.DataFrame],
    quantile: float = 0.10,
    ids: Optional[pd.Index] = None,
) -> Tuple[Dict[str, float], pd.Index]:
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile muss in (0,1) liegen.")

    aligned, used_ids = _coerce_and_slice_matrices(matrices, ids=ids)

    thr: Dict[str, float] = {}
    for name, D_df in aligned.items():
        du = _upper_triangle(D_df.to_numpy())
        thr[name] = float(np.quantile(du, quantile))

    return thr, used_ids


def _align_y_to_used_ids(used_ids: pd.Index, y: pd.Series) -> np.ndarray:
    if not isinstance(y, pd.Series):
        raise ValueError("y muss pd.Series mit Index=Customer IDs sein.")

    missing = set(used_ids) - set(y.index)
    if missing:
        ex = list(sorted(missing))[:5]
        more = " ..." if len(missing) > 5 else ""
        raise ValueError(f"y enthält nicht alle used_ids. Missing: {ex}{more}")

    return y.loc[used_ids].to_numpy(dtype=int)


def pairwise_agreement_disagreement(
    matrices: Dict[str, pd.DataFrame],
    quantile: float = 0.10,
    ids: Optional[pd.Index] = None,
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Index]:
    aligned, used_ids = _coerce_and_slice_matrices(matrices, ids=ids)
    thresholds, _ = thresholds_by_quantile(aligned, quantile=quantile, ids=used_ids)

    names = list(aligned.keys())
    iu = np.triu_indices(len(used_ids), k=1)

    same_mask = None
    diff_mask = None
    if y is not None:
        y_arr = _align_y_to_used_ids(used_ids, y)
        same_mask = (y_arr[iu[0]] == y_arr[iu[1]])
        diff_mask = ~same_mask

    rows = []

    def compute_stats(
        a_sim: np.ndarray,
        b_sim: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Dict[str, float]:
        if mask is None:
            m = slice(None)
            n_pairs = int(a_sim.shape[0])
        else:
            if mask.sum() == 0:
                return {
                    "agree_similar": np.nan,
                    "agree_dissimilar": np.nan,
                    "A_only_similar": np.nan,
                    "B_only_similar": np.nan,
                    "n_pairs": 0,
                }
            m = mask
            n_pairs = int(mask.sum())

        agree_sim = float(np.mean(a_sim[m] & b_sim[m]))
        agree_dis = float(np.mean(~a_sim[m] & ~b_sim[m]))
        A_only = float(np.mean(a_sim[m] & ~b_sim[m]))
        B_only = float(np.mean(~a_sim[m] & b_sim[m]))

        return {
            "agree_similar": agree_sim,
            "agree_dissimilar": agree_dis,
            "A_only_similar": A_only,
            "B_only_similar": B_only,
            "n_pairs": n_pairs,
        }

    for i, A in enumerate(names):
        DA = aligned[A].to_numpy()
        a = DA[iu]
        tA = thresholds[A]
        a_sim = (a <= tA)

        for j, B in enumerate(names):
            if j <= i:
                continue

            DB = aligned[B].to_numpy()
            b = DB[iu]
            tB = thresholds[B]
            b_sim = (b <= tB)

            st_all = compute_stats(a_sim, b_sim, None)
            rows.append({
                "A": A,
                "B": B,
                "subset": "all",
                "quantile": quantile,
                "tA": tA,
                "tB": tB,
                **st_all,
            })

            if y is not None:
                st_same = compute_stats(a_sim, b_sim, same_mask)
                rows.append({
                    "A": A,
                    "B": B,
                    "subset": "same",
                    "quantile": quantile,
                    "tA": tA,
                    "tB": tB,
                    **st_same,
                })

                st_diff = compute_stats(a_sim, b_sim, diff_mask)
                rows.append({
                    "A": A,
                    "B": B,
                    "subset": "diff",
                    "quantile": quantile,
                    "tA": tA,
                    "tB": tB,
                    **st_diff,
                })

    summary_df = pd.DataFrame(rows).set_index(["A", "B", "subset"]).sort_index()
    return summary_df, thresholds, used_ids


def disagreement_score(summary_df: pd.DataFrame, subset: str = "all") -> pd.Series:
    df = summary_df.xs(subset, level="subset")
    return (df["A_only_similar"] + df["B_only_similar"]).sort_values(ascending=False)
