"""
repeated_oof.py — Neuerung 2, built on the Stage-8 motor.

A repeat is just a different fold layout, i.e. a different random_state. Because
the seed already lives in the config_key (via run_signature), running R repeats of
the winner configuration is nothing but R * n_splits more cached, resumable fold
jobs — no new machinery. Each customer lands in exactly one validation fold per
repeat, so it gets R predictions.

Aggregation reads the fold rows back from the cache and uses the ``va_idx_json``
column (option A) to scatter each fold's y_pred to the right customers:

    correct_rate_i = (# repeats predicting i correctly) / (# repeats i was valid)

which is then linkable to the structural purity_i / margin_i (§4 Verknüpfung).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine
from .search import multi_key, best_configs
from .models import M3GLVQ_Global, M3GLVQ_Label

_MODEL_CLS = {"M3GLVQ_Label": M3GLVQ_Label, "M3GLVQ_Global": M3GLVQ_Global}


def _winner_jobs_for_repeat(model_name, cfg, T, n_splits, rs, base_seed):
    """One repeat = one multi job for the winner (K0,K1,eta) under fold-layout rs."""
    cls = _MODEL_CLS[model_name]
    key = multi_key(model_name, cfg["K0"], cfg["K1"], cfg["eta"], T, n_splits, rs, base_seed)
    return {
        "key": key, "kind": "multi", "model": model_name, "cls": cls,
        "K0": int(cfg["K0"]), "K1": int(cfg["K1"]), "eta": float(cfg["eta"]),
        "T": int(T), "n_splits": int(n_splits),
    }


def run_repeated_oof(best_per_model, DL, y, cache_db, *,
                     T=150, n_splits=10, n_repeats=5,
                     random_state0=1000, base_seed=42, n_jobs=2):
    """Run R repeats of each winner config; every repeat reseeds the fold layout.

    Returns {model_name: {"config_keys": [...], "cfg": cfg, "T":T, "n_splits":n_splits}}
    so the aggregation step knows exactly which cached configs to read.
    """
    manifest = {}
    for model_name, cfg in best_per_model.items():
        keys = []
        for r in range(n_repeats):
            rs = random_state0 + r
            job = _winner_jobs_for_repeat(model_name, cfg, T, n_splits, rs, base_seed)
            # each repeat has its own fold layout -> pass rs as random_state
            engine.run_jobs([job], DL, y, n_splits, cache_db, n_jobs=n_jobs,
                            base_seed=base_seed, random_state=rs,
                            stage_name=f"rOOF::{model_name}::rep{r}")
            keys.append(job["key"])
        manifest[model_name] = {"config_keys": keys, "cfg": cfg, "T": T,
                                "n_splits": n_splits, "n_repeats": n_repeats}
    return manifest


def aggregate_correctness(cache_db, config_keys, m):
    """Scatter cached predictions back to customers via va_idx and count hits.

    Returns dict with length-m arrays: correct_rate, n_valid, pred_majority,
    and a per-customer 2-column vote matrix.
    """
    hits = np.zeros(m); n = np.zeros(m); votes = np.zeros((m, 2))

    for ck in config_keys:
        cached = engine.load_fold_cache(cache_db, ck)
        for fold, c in cached.items():
            if c["status"] != "ok" or c["va_idx"] is None:
                continue  # degenerate fold or legacy row without va_idx
            va = np.asarray(c["va_idx"], dtype=int)
            yp = np.asarray(c["y_pred"], dtype=int)
            yt = np.asarray(c["y_true"], dtype=int)
            n[va] += 1.0
            hits[va] += (yp == yt).astype(float)
            for cls in (0, 1):
                votes[va[yp == cls], cls] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, hits / n, np.nan)
    return {"correct_rate": rate, "n_valid": n.astype(int),
            "pred_majority": votes.argmax(1), "votes": votes}


def correctness_frame(codes, y, correctness):
    """Tidy per-customer correctness table."""
    return pd.DataFrame({
        "customer_id": np.asarray(codes, dtype=str),
        "y": np.asarray(y, dtype=int),
        "correct_rate": correctness["correct_rate"],
        "n_valid": correctness["n_valid"],
        "pred_majority": correctness["pred_majority"].astype(int),
    })


def link_correctness_to_structure(correct_df, pointwise_df, on="customer_id"):
    """§4 Verknüpfung: join repeated-OOF correct_rate_i with structural
    purity_i / margin_i (per view + dominant view). Both frames keyed by
    customer_id (str)."""
    a = correct_df.copy(); b = pointwise_df.copy()
    a[on] = a[on].astype(str); b[on] = b[on].astype(str)
    return a.merge(b, on=on, how="inner")


def run_and_link(best_per_model, DL, y, codes, pointwise_df, cache_db, *,
                 T=150, n_splits=10, n_repeats=5, random_state0=1000,
                 base_seed=42, n_jobs=2):
    """Convenience: run repeated-OOF for all winners, aggregate, and link to
    structure. Returns {model_name: linked_df}."""
    manifest = run_repeated_oof(best_per_model, DL, y, cache_db, T=T, n_splits=n_splits,
                                n_repeats=n_repeats, random_state0=random_state0,
                                base_seed=base_seed, n_jobs=n_jobs)
    m = len(y)
    out = {}
    for model_name, info in manifest.items():
        corr = aggregate_correctness(cache_db, info["config_keys"], m)
        cframe = correctness_frame(codes, y, corr)
        out[model_name] = link_correctness_to_structure(cframe, pointwise_df)
    return out, manifest
