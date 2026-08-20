"""
search.py — Stage-8 search on top of engine.run_jobs.

Grid over the three model families (single-view MGLVQ, corrected Global, Label),
run in parallel with fold-level caching/resume, selecting the best (K, eta) per
model type. Ported from the optimized Stage-8 notebook but parameterized: no
module globals, config keys built via engine.run_signature so the seed is part
of the key (which is what makes repeated-OOF a trivial reseed later).

Config-key builders (multi_key / single_key) are shared with repeated_oof.py so
the two never drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine
from .models import M3GLVQ_Global, M3GLVQ_Label
from .single_view_mglvq.mglvq_fast import MGLVQ

VIEW_TAGS = ("naics", "hs", "am")  # lowercase to match structural / paper_tables


# --------------------------------------------------------------------- key builders
def multi_key(model_name, K0, K1, eta, T, n_splits, random_state, base_seed):
    sig = engine.run_signature(T, n_splits, random_state, base_seed)
    return f"{model_name}|K{K0}/{K1}|eta{eta}|{sig}"


def single_key(tag, K, T, n_splits, random_state, base_seed):
    sig = engine.run_signature(T, n_splits, random_state, base_seed)
    return f"single::{tag}|K{K}|{sig}"


# ------------------------------------------------------------------------- job build
def build_multi_jobs(k_pairs, etas, T, n_splits, random_state, base_seed,
                     run_label=True, run_global=True, model_filter=None):
    """Label/Global jobs. model_filter (dict name->set of (K0,K1,eta)) restricts
    each model to its own candidate set (used by staged refinement)."""
    model_defs = []
    if run_label:
        model_defs.append(("M3GLVQ_Label", M3GLVQ_Label))
    if run_global:
        model_defs.append(("M3GLVQ_Global", M3GLVQ_Global))

    jobs = []
    for name, cls in model_defs:
        if model_filter is None:
            candidates = [(k0, k1, eta) for (k0, k1) in k_pairs for eta in etas]
        else:
            candidates = sorted(model_filter.get(name, set()))
        for K0, K1, eta in candidates:
            jobs.append({
                "key": multi_key(name, K0, K1, eta, T, n_splits, random_state, base_seed),
                "kind": "multi", "model": name, "cls": cls,
                "K0": int(K0), "K1": int(K1), "eta": float(eta),
                "T": int(T), "n_splits": int(n_splits),
            })
    jobs.sort(key=lambda j: (j["K0"], j["K1"], j["eta"], j["model"]))
    return jobs


def build_single_jobs(k_values, T, n_splits, random_state, base_seed, view_tags=VIEW_TAGS):
    jobs = []
    for view_idx, tag in enumerate(view_tags):
        for K in k_values:
            jobs.append({
                "key": single_key(tag, K, T, n_splits, random_state, base_seed),
                "kind": "single", "model": f"single::{tag}", "cls": MGLVQ,
                "view_idx": int(view_idx), "tag": tag,
                "K": int(K), "T": int(T), "n_splits": int(n_splits),
            })
    return jobs


# --------------------------------------------------------------------- winner select
def top_per_model(df, top_n, metric="bal_acc"):
    """Select winners independently for Label and Global."""
    pieces = []
    for model in ("M3GLVQ_Label", "M3GLVQ_Global"):
        if "model" not in df.columns or model not in set(df["model"]):
            continue
        d = df[(df["model"] == model) & (df["status"] == "ok")].copy()
        d = d.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)
        if len(d):
            pieces.append(d)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _parse_k_pair(k_text):
    k0, k1 = str(k_text).split("/")
    return int(k0), int(k1)


def _eta_neighbors(eta, eta_grid, radius=1):
    arr = np.asarray(eta_grid, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(eta))))
    lo = max(0, idx - radius); hi = min(len(arr), idx + radius + 1)
    return [float(x) for x in arr[lo:hi]]


def refine_candidates_from_winners(winners, eta_grid, k_radius=1, eta_radius=1,
                                   k_min=3, k_max=10):
    out = {}
    if winners.empty:
        return out
    for _, r in winners.iterrows():
        model = r["model"]; K0, K1 = _parse_k_pair(r["K"]); eta = float(r["eta"])
        out.setdefault(model, set())
        k0_vals = range(max(k_min, K0 - k_radius), min(k_max, K0 + k_radius) + 1)
        k1_vals = range(max(k_min, K1 - k_radius), min(k_max, K1 + k_radius) + 1)
        e_vals = _eta_neighbors(eta, eta_grid, eta_radius)
        for k0 in k0_vals:
            for k1 in k1_vals:
                for e in e_vals:
                    out[model].add((int(k0), int(k1), float(e)))
    return out


def finalists_from_winners(winners):
    out = {}
    if winners.empty:
        return out
    for _, r in winners.iterrows():
        model = r["model"]; K0, K1 = _parse_k_pair(r["K"])
        out.setdefault(model, set()).add((K0, K1, float(r["eta"])))
    return out


def best_configs(results_df, metric="bal_acc"):
    """Return {model_name: dict(K0,K1,eta,T,n_splits,metric)} for the top row per
    multi-view model, ready to hand to repeated_oof."""
    out = {}
    for model in ("M3GLVQ_Label", "M3GLVQ_Global"):
        d = results_df[(results_df["model"] == model) & (results_df["status"] == "ok")]
        d = d.dropna(subset=[metric])
        if d.empty:
            continue
        r = d.sort_values(metric, ascending=False).iloc[0]
        K0, K1 = _parse_k_pair(r["K"])
        out[model] = {"K0": K0, "K1": K1, "eta": float(r["eta"]), metric: float(r[metric])}
    return out


# ------------------------------------------------------------------------- drivers
# Default grids (mirror the Stage-8 presets).
FULL_K_VALUES = list(range(5, 14))
FULL_K_PAIRS = [(k0, k1) for k0 in FULL_K_VALUES for k1 in FULL_K_VALUES]
FULL_ETAS = [0.003, 0.0075, 0.01, 0.015, 0.025, 0.03, 0.04, 0.05]
FULL_SINGLE_VIEW_K = list(range(3, 11))

STAGE1_K_VALUES = [3, 6, 9]
STAGE1_K_PAIRS = [(k0, k1) for k0 in STAGE1_K_VALUES for k1 in STAGE1_K_VALUES]
STAGE1_ETAS = [0.003, 0.015, 0.03, 0.05]


def full_search(DL, y, cache_db, *, T=150, n_splits=3, random_state=42, base_seed=42,
                n_jobs=2, run_single=True, run_global=True, run_label=True,
                single_k=None, k_pairs=None, etas=None, metric="bal_acc"):
    """Single exhaustive grid at one fidelity. Returns (results_df, best_per_model)."""
    single_k = FULL_SINGLE_VIEW_K if single_k is None else single_k
    k_pairs = FULL_K_PAIRS if k_pairs is None else k_pairs
    etas = FULL_ETAS if etas is None else etas

    jobs = []
    if run_single:
        jobs += build_single_jobs(single_k, T, n_splits, random_state, base_seed)
    if run_global or run_label:
        jobs += build_multi_jobs(k_pairs, etas, T, n_splits, random_state, base_seed,
                                 run_label=run_label, run_global=run_global)

    df = engine.run_jobs(jobs, DL, y, n_splits, cache_db, n_jobs=n_jobs,
                         base_seed=base_seed, random_state=random_state, stage_name="full")
    return df, best_configs(df, metric)


def staged_search(DL, y, cache_db, *, random_state=42, base_seed=42, n_jobs=2,
                  metric="bal_acc",
                  stage1=dict(T=100, n_splits=3, top=6),
                  stage2=dict(T=100, n_splits=5, k_radius=1, eta_radius=1, top=3),
                  stage3=dict(T=150, n_splits=10),
                  run_single=True, run_global=True, run_label=True,
                  single_k=None):
    """Coarse -> refine -> finalists at paper fidelity. Returns (final_df, best_per_model).

    Single-view baselines (if enabled) are evaluated once at Stage-3 fidelity.
    """
    hist = {}

    # Stage 1: sparse grid, low fidelity
    s1 = build_multi_jobs(STAGE1_K_PAIRS, STAGE1_ETAS, stage1["T"], stage1["n_splits"],
                          random_state, base_seed, run_label=run_label, run_global=run_global)
    df1 = engine.run_jobs(s1, DL, y, stage1["n_splits"], cache_db, n_jobs=n_jobs,
                          base_seed=base_seed, random_state=random_state, stage_name="stage1")
    hist["stage1"] = df1
    w1 = top_per_model(df1, stage1["top"], metric)

    # Stage 2: neighbourhoods around winners, medium fidelity
    filt2 = refine_candidates_from_winners(w1, FULL_ETAS,
                                           k_radius=stage2["k_radius"], eta_radius=stage2["eta_radius"])
    s2 = build_multi_jobs(None, None, stage2["T"], stage2["n_splits"], random_state, base_seed,
                          run_label=run_label, run_global=run_global, model_filter=filt2)
    df2 = engine.run_jobs(s2, DL, y, stage2["n_splits"], cache_db, n_jobs=n_jobs,
                          base_seed=base_seed, random_state=random_state, stage_name="stage2")
    hist["stage2"] = df2
    w2 = top_per_model(df2, stage2["top"], metric)

    # Stage 3: exact finalists at paper fidelity
    filt3 = finalists_from_winners(w2)
    s3 = build_multi_jobs(None, None, stage3["T"], stage3["n_splits"], random_state, base_seed,
                          run_label=run_label, run_global=run_global, model_filter=filt3)
    if run_single:
        s3 += build_single_jobs(FULL_SINGLE_VIEW_K if single_k is None else single_k,
                                stage3["T"], stage3["n_splits"], random_state, base_seed)
    df3 = engine.run_jobs(s3, DL, y, stage3["n_splits"], cache_db, n_jobs=n_jobs,
                          base_seed=base_seed, random_state=random_state, stage_name="stage3")
    hist["stage3"] = df3

    final = pd.concat([d for d in hist.values() if not d.empty], ignore_index=True)
    return df3, best_configs(df3, metric), hist
