"""
single_view_global.py — single-view baseline derived from the corrected GLOBAL
model instead of the standalone mglvq_fast.

A single view is the global model with the weights frozen to a one-hot mixture and
eta=0: D* = Sigma a_v^2 D_v collapses to one view, prototypes are placed by the
SAME hard-RNG init + RF-scan swap as the multi-view arms, and the buggy separate
medoid-swap model drops out. The only difference across all three arms becomes the
weighting (one-hot vs. learned).

Full asymmetric K grid: K0 (class 0) and K1 (class 1) vary independently, exactly
like the multi-view search (product(K0, K1)). Per-class prototype collapse is
handled per class, so asymmetric pairs often survive on tie-heavy views (HS/AM)
even when the symmetric high-K ones do not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from . import engine
from .models import M3GLVQ_Global

VIEW_TAGS = ("naics", "hs", "am")

# full symmetric+asymmetric grid, same range as the paper multi-view grid
DEFAULT_K_PAIRS = [(k0, k1) for k0 in range(3, 11) for k1 in range(3, 11)]


def _one_hot(view_idx, n=3):
    v = np.zeros(n); v[view_idx] = 1.0
    return v


def _sv_key(tag, K0, K1, T, n_splits, random_state, base_seed):
    sig = engine.run_signature(T, n_splits, random_state, base_seed)
    return f"svglobal::{tag}|K{K0}/{K1}|{sig}"


def _build_jobs(k_pairs, T, n_splits, random_state, base_seed, view_tags=VIEW_TAGS):
    jobs = []
    for vi, tag in enumerate(view_tags):
        v_init = np.sqrt(_one_hot(vi))          # normalize_l2(sqrt(one-hot))**2 == one-hot
        for K0, K1 in k_pairs:
            jobs.append({
                "key": _sv_key(tag, K0, K1, T, n_splits, random_state, base_seed),
                "kind": "fixed", "model": f"svglobal::{tag}", "cls": M3GLVQ_Global,
                "K0": int(K0), "K1": int(K1), "eta": 0.0, "v_init": v_init,
                "T": int(T), "n_splits": int(n_splits),
                "tag": tag,
            })
    return jobs


def _oof_from_cache(cache_db, key, m):
    """Reconstruct per-customer OOF prediction for one config via va_idx.
    Returns None if the config has any degenerate/failed fold (per-class collapse)."""
    yp = np.full(m, -1, dtype=int)
    cached = engine.load_fold_cache(cache_db, key)
    if not cached:
        return None
    for c in cached.values():
        if c["status"] != "ok" or c["va_idx"] is None:
            return None                          # collapsed / incomplete config -> skip
        va = np.asarray(c["va_idx"], dtype=int)
        yp[va] = np.asarray(c["y_pred"], dtype=int)
    return yp


def run_single_view_from_global(DL, y, codes, cache_db, *,
                                k_pairs=DEFAULT_K_PAIRS,
                                T=150, n_splits=10, random_state=42, base_seed=42,
                                n_jobs=2, view_tags=VIEW_TAGS, progress="print"):
    """Grid the three one-hot views over the full (K0,K1) grid, pick the best pair
    per view by OOF balanced accuracy, and return (summary_df, correct_wide).

    summary_df columns (superset of build_single_view_baseline_table):
        matrix, best_K ("K0/K1"), best_K0, best_K1,
        balanced_accuracy_from_oof, recall_from_oof, tp, fp, tn, fn,
        n_pairs_valid, n_pairs_total
    correct_wide columns: customer_id, y, correct_naics, correct_hs, correct_am
    """
    m = len(y); y = np.asarray(y, dtype=int); codes = np.asarray(codes, dtype=str)
    jobs = _build_jobs(k_pairs, T, n_splits, random_state, base_seed, view_tags)
    engine.run_jobs(jobs, DL, y, n_splits, cache_db, n_jobs=n_jobs,
                    base_seed=base_seed, random_state=random_state,
                    stage_name="svglobal", progress=progress)

    summary_rows = []
    best_pred = {}
    for vi, tag in enumerate(view_tags):
        best = None; n_valid = 0
        for K0, K1 in k_pairs:
            key = _sv_key(tag, K0, K1, T, n_splits, random_state, base_seed)
            yp = _oof_from_cache(cache_db, key, m)
            if yp is None:
                continue                          # collapsed pair -> not comparable
            ok = yp != -1
            if ok.sum() == 0:
                continue
            n_valid += 1
            ba = balanced_accuracy_score(y[ok], yp[ok])
            if best is None or ba > best["ba"]:
                best = {"K0": K0, "K1": K1, "ba": ba, "yp": yp, "ok": ok}
        if best is None:
            summary_rows.append({"matrix": tag, "best_K": None, "best_K0": None,
                                 "best_K1": None, "balanced_accuracy_from_oof": np.nan,
                                 "recall_from_oof": np.nan, "tp": 0, "fp": 0, "tn": 0,
                                 "fn": 0, "n_pairs_valid": 0, "n_pairs_total": len(k_pairs)})
            continue
        yp, ok = best["yp"], best["ok"]
        rec = recall_score(y[ok], yp[ok], pos_label=1, zero_division=0)
        tp = int(((yp == 1) & (y == 1) & ok).sum()); fp = int(((yp == 1) & (y == 0) & ok).sum())
        tn = int(((yp == 0) & (y == 0) & ok).sum()); fn = int(((yp == 0) & (y == 1) & ok).sum())
        summary_rows.append({
            "matrix": tag,
            "best_K": f"{best['K0']}/{best['K1']}", "best_K0": best["K0"], "best_K1": best["K1"],
            "balanced_accuracy_from_oof": best["ba"], "recall_from_oof": rec,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_pairs_valid": n_valid, "n_pairs_total": len(k_pairs),
        })
        best_pred[tag] = yp

    summary_df = pd.DataFrame(summary_rows)

    correct = {"customer_id": codes, "y": y}
    for tag in view_tags:
        if tag in best_pred:
            yp = best_pred[tag]
            correct[f"correct_{tag}"] = np.where(yp != -1, (yp == y).astype(int), np.nan)
    correct_wide = pd.DataFrame(correct)

    return summary_df, correct_wide