"""
weight_profiles.py — the multi-view weight-profile clustering + fixed-weight
profile suite (old Block-3 tail + paper Table 9), rebuilt on the Stage-8 motor.

The old pipeline re-ran CV serially just to capture per-fold weights. Here the
search already cached them (``vweights_json``), so we:

  1. pull the per-fold Global weight vectors (a^2 mixtures) straight from SQLite,
  2. cluster them (KMeans on the a^2 simplex, core-filtered centroids),
  3. turn cluster centroids + canonical single-view corners into fixed profiles,
  4. evaluate each fixed profile as an engine job with ``eta=0`` and a frozen
     ``v_init`` — with the crucial parametrization fix: the corrected model uses
     the squared parametrization (Sigma a_v^2 D_v) and L2-normalizes v_init, so a
     mixture ``m`` (sum 1) must be injected as ``v_init = sqrt(m)`` to make the
     effective a^2 equal ``m``.
  5. aggregate per-customer correctness via ``va_idx`` -> ``fixed_oof_df`` for
     ``paper_tables.build_fixed_profile_dominance_table`` (Table 9).

Pure numpy/pandas/sklearn; no model import except through the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.cluster import KMeans

from . import engine
from .models import M3GLVQ_Global

V_COLS = ("vweight_0", "vweight_1", "vweight_2")


# ------------------------------------------------------- collect weights from cache
def collect_fold_weights(cache_db, model="M3GLVQ_Global"):
    """One row per (config, fold) with the fold's a^2 weights + per-fold bal_acc.

    Reads the search cache directly, so nothing is recomputed.
    """
    import sqlite3, json
    rows = []
    with sqlite3.connect(cache_db, timeout=60) as con:
        cur = con.execute(
            """SELECT config_key, fold, k_value, eta, status,
                      y_true_json, y_pred_json, vweights_json
               FROM fold_results WHERE model = ?""",
            (model,),
        )
        for ck, fold, kval, eta, status, yt, yp, vw in cur.fetchall():
            if status != "ok" or vw is None:
                continue
            w = json.loads(vw)
            if w.get("kind") != "global":
                continue
            a_sq = np.asarray(w["a_sq"], dtype=float)
            if a_sq.size != 3:
                continue
            try:
                k0, k1 = str(kval).split("/")
            except ValueError:
                k0, k1 = np.nan, np.nan
            ba = np.nan
            if yt is not None and yp is not None:
                yt_a = np.asarray(json.loads(yt)); yp_a = np.asarray(json.loads(yp))
                if yt_a.size:
                    ba = balanced_accuracy_score(yt_a, yp_a)
            rows.append({
                "config_key": ck, "fold": int(fold),
                "K_0": float(k0), "K_1": float(k1), "eta": eta,
                "vweight_0": a_sq[0], "vweight_1": a_sq[1], "vweight_2": a_sq[2],
                "balanced_accuracy": ba,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- clustering
def select_top_runs(fold_weights, metric_col="balanced_accuracy", quantile=0.9):
    thr = fold_weights[metric_col].quantile(quantile)
    return fold_weights[fold_weights[metric_col] >= thr].copy(), float(thr)


def cluster_weight_profiles(df_high, n_clusters=5, random_state=42, top_frac=0.20,
                            v_cols=V_COLS):
    """KMeans on a^2 vectors; centroids re-estimated on the top_frac core of each
    cluster (robust to boundary folds). Returns filtered centroids + labeled df."""
    X = df_high[list(v_cols)].to_numpy(dtype=float)
    n_clusters = int(min(n_clusters, X.shape[0]))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    core = np.zeros(len(df_high), dtype=bool)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        d = np.linalg.norm(X[idx] - centers[c], axis=1)
        n_top = max(1, int(len(idx) * top_frac))
        core[idx[np.argsort(d)[:n_top]]] = True

    filtered = np.zeros_like(centers)
    for c in range(n_clusters):
        mask = (labels == c) & core
        filtered[c] = X[mask].mean(0) if mask.any() else centers[c]

    out = df_high.copy().reset_index(drop=True)
    out["cluster"] = labels
    return {"df": out, "labels": labels, "centers": centers,
            "filtered_centers": filtered, "core_mask": core, "n_clusters": n_clusters}


def cluster_summary(cluster_res, v_cols=V_COLS):
    df = cluster_res["df"]; fc = cluster_res["filtered_centers"]
    rows = []
    for c in sorted(df["cluster"].unique()):
        g = df[df["cluster"] == c]
        rows.append({
            "cluster": int(c), "size": int(len(g)),
            "mean_balanced_accuracy": float(g["balanced_accuracy"].mean()),
            "vweight_0": float(fc[c][0]), "vweight_1": float(fc[c][1]), "vweight_2": float(fc[c][2]),
            "dominant_view": ("naics", "hs", "am")[int(np.argmax(fc[c]))],
        })
    return pd.DataFrame(sorted(rows, key=lambda r: r["size"], reverse=True))


# ------------------------------------------------------------------ fixed profiles
def _normalize_mixture(w):
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(3, 1.0 / 3)


def build_fixed_profiles(cluster_res=None, K=(6, 6), include_canonical=True,
                         include_uniform=True):
    """Fixed profiles as a^2 mixtures (sum 1). Canonical single-view corners +
    uniform + one 'primary' profile per cluster centroid."""
    profiles = []
    K = (int(K[0]), int(K[1]))
    if include_canonical:
        profiles += [
            {"profile_name": "naics_only", "weights": np.array([1.0, 0.0, 0.0]), "K": K},
            {"profile_name": "hs_only",    "weights": np.array([0.0, 1.0, 0.0]), "K": K},
            {"profile_name": "am_only",    "weights": np.array([0.0, 0.0, 1.0]), "K": K},
        ]
    if include_uniform:
        profiles.append({"profile_name": "uniform",
                         "weights": np.array([1/3, 1/3, 1/3]), "K": K})
    if cluster_res is not None:
        summ = cluster_summary(cluster_res)
        for _, r in summ.iterrows():
            profiles.append({
                "profile_name": f"C{int(r['cluster'])}_primary",
                "weights": _normalize_mixture([r["vweight_0"], r["vweight_1"], r["vweight_2"]]),
                "K": K,
            })
    return profiles


# ------------------------------------------------------------ fixed-weight running
def fixed_key(profile_name, K0, K1, T, n_splits, random_state, base_seed):
    sig = engine.run_signature(T, n_splits, random_state, base_seed)
    return f"fixed::{profile_name}|K{K0}/{K1}|{sig}"


def build_fixed_jobs(profiles, T, n_splits, random_state, base_seed):
    """One engine job per profile: eta=0, v_init=sqrt(mixture) (squared-param fix)."""
    jobs = []
    for p in profiles:
        K0, K1 = int(p["K"][0]), int(p["K"][1])
        mix = _normalize_mixture(p["weights"])
        v_init = np.sqrt(mix)                 # normalize_l2(sqrt(mix))**2 == mix
        jobs.append({
            "key": fixed_key(p["profile_name"], K0, K1, T, n_splits, random_state, base_seed),
            "kind": "fixed", "model": p["profile_name"], "cls": M3GLVQ_Global,
            "K0": K0, "K1": K1, "eta": 0.0, "v_init": v_init,
            "T": int(T), "n_splits": int(n_splits),
            "profile_name": p["profile_name"],
        })
    return jobs


def run_fixed_profiles(profiles, DL, y, codes, cache_db, *,
                       T=150, n_splits=10, random_state=42, base_seed=42, n_jobs=2):
    """Evaluate all fixed profiles (frozen weights) and return a long fixed_oof_df
    with per-customer correctness for Table 9."""
    jobs = build_fixed_jobs(profiles, T, n_splits, random_state, base_seed)
    engine.run_jobs(jobs, DL, y, n_splits, cache_db, n_jobs=n_jobs,
                    base_seed=base_seed, random_state=random_state, stage_name="fixed")

    codes = np.asarray(codes, dtype=str); y = np.asarray(y, dtype=int)
    records = []
    for job in jobs:
        cached = engine.load_fold_cache(cache_db, job["key"])
        for fold, c in cached.items():
            if c["status"] != "ok" or c["va_idx"] is None:
                continue
            va = np.asarray(c["va_idx"], dtype=int)
            yp = np.asarray(c["y_pred"], dtype=int)
            yt = np.asarray(c["y_true"], dtype=int)
            for j, cust in enumerate(va):
                records.append({
                    "customer_id": codes[cust],
                    "profile_name": job["profile_name"],
                    "K_0": job["K0"], "K_1": job["K1"],
                    "y_true": int(yt[j]), "y_pred": int(yp[j]),
                    "correct": int(yp[j] == yt[j]),
                })
    return pd.DataFrame(records)


def default_profile_map(profiles):
    """Map display names -> profile_name for build_fixed_profile_dominance_table."""
    return {p["profile_name"]: p["profile_name"] for p in profiles}
