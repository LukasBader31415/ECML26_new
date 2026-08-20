"""
engine.py — the Stage-8 search/cache motor, decoupled from the notebook globals.

Ported byte-faithfully from ecml26_pipeline_optimized_stage8_weights with ONE
addition and one refactor:

  * ADDED column ``va_idx_json`` to ``fold_results`` (option A): each cached fold
    row now records *which* customer indices its predictions belong to. This makes
    rows self-describing and is what the repeated-OOF aggregation needs to map
    ``y_pred`` back to individual customers (correct_rate_i) and to link them to
    the structural ``purity_i`` / ``margin_i``. The migration follows the existing
    ALTER-TABLE pattern, so pre-existing caches upgrade in place (old rows get
    va_idx = NULL and can be regenerated on next fill).

  * REFACTORED ``run_stage`` -> ``run_jobs``: takes DL, y, n_splits, cache_db,
    n_jobs, seeds as explicit arguments instead of reading module globals.

Job building (build_multi_jobs / build_single_jobs / staged search) lives in
search.py (Phase 3). This module is the cache + harness + parallel runner only.
"""
from __future__ import annotations

import gc
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .models.proto_init import DegenerateInitError

# --------------------------------------------------------------------------- folds
def make_splits(y, n_splits, random_state):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))


def prepare_folds(DL, y, n_splits, random_state):
    """Precompute all view-specific train/validation matrices once for a stage.

    Each entry also carries ``va`` (the validation customer indices), which the
    harness now persists so predictions stay attributable per customer.
    """
    splits = make_splits(y, n_splits, random_state)
    prepared = []
    for fold, (tr, va) in enumerate(splits, start=1):
        prepared.append({
            "fold": fold,
            "tr": tr,
            "va": va,
            "y_tr": y[tr],
            "y_va": y[va],
            "DL_tr": [D[np.ix_(tr, tr)] for D in DL],
            "DL_va": [D[np.ix_(va, tr)] for D in DL],
        })
    return prepared


# ------------------------------------------------------------------------ cache DB
def init_cache_db(path):
    with sqlite3.connect(path, timeout=60) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("""
            CREATE TABLE IF NOT EXISTS fold_results (
                config_key TEXT NOT NULL,
                fold INTEGER NOT NULL,
                model TEXT NOT NULL,
                k_value TEXT,
                eta REAL,
                status TEXT NOT NULL,
                note TEXT,
                y_true_json TEXT,
                y_pred_json TEXT,
                va_idx_json TEXT,
                vweights_json TEXT,
                protos_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (config_key, fold)
            )
        """)
        # in-place migration for pre-existing caches
        for _col in ("va_idx_json TEXT", "vweights_json TEXT", "protos_json TEXT"):
            try:
                con.execute(f"ALTER TABLE fold_results ADD COLUMN {_col}")
            except sqlite3.OperationalError:
                pass  # column already present
        con.commit()


def load_fold_cache(path, config_key):
    """Return cached folds for one configuration, keyed by fold number."""
    if not Path(path).exists():
        return {}
    with sqlite3.connect(path, timeout=60) as con:
        rows = con.execute(
            """
            SELECT fold, model, k_value, eta, status, note,
                   y_true_json, y_pred_json, va_idx_json, vweights_json, protos_json
            FROM fold_results
            WHERE config_key = ?
            ORDER BY fold
            """,
            (config_key,),
        ).fetchall()

    out = {}
    for (fold, model, k_value, eta, status, note,
         yt_json, yp_json, va_json, vw_json, pr_json) in rows:
        out[int(fold)] = {
            "model": model,
            "K": k_value,
            "eta": eta,
            "status": status,
            "note": note or "",
            "y_true": None if yt_json is None else np.asarray(json.loads(yt_json), dtype=int),
            "y_pred": None if yp_json is None else np.asarray(json.loads(yp_json), dtype=int),
            "va_idx": None if va_json is None else np.asarray(json.loads(va_json), dtype=int),
            "vweights": None if vw_json is None else json.loads(vw_json),
            "protos": None if pr_json is None else json.loads(pr_json),
        }
    return out


def save_fold_cache(path, config_key, fold, model, K, eta, status, note,
                    y_true, y_pred, va_idx=None, vweights=None, protos=None):
    """Concurrent-safe fold write. SQLite WAL serializes the tiny writes."""
    payload = (
        config_key,
        int(fold),
        str(model),
        None if K is None else str(K),
        None if eta is None else float(eta),
        str(status),
        str(note or ""),
        None if y_true is None else json.dumps(np.asarray(y_true, dtype=int).tolist()),
        None if y_pred is None else json.dumps(np.asarray(y_pred, dtype=int).tolist()),
        None if va_idx is None else json.dumps(np.asarray(va_idx, dtype=int).tolist()),
        None if vweights is None else json.dumps(vweights),
        None if protos is None else json.dumps(list(protos)),
        datetime.now(timezone.utc).isoformat(),
    )

    last_error = None
    for attempt in range(8):
        try:
            with sqlite3.connect(path, timeout=60) as con:
                con.execute(
                    """
                    INSERT OR REPLACE INTO fold_results
                    (config_key, fold, model, k_value, eta, status, note,
                     y_true_json, y_pred_json, va_idx_json, vweights_json, protos_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                con.commit()
            return
        except sqlite3.OperationalError as e:
            last_error = e
            if "locked" not in str(e).lower():
                raise
            time.sleep(0.25 * (attempt + 1))
    raise last_error


# ---------------------------------------------------------------------- CV harness
def _metrics(yt, yp):
    return {
        "acc": accuracy_score(yt, yp),
        "bal_acc": balanced_accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, average="macro"),
    }


def extract_vweights(clf):
    """Final a^2 weights: Global -> {'kind':'global','a_sq':[...]},
    Label -> {'kind':'label','labels':[...],'a_sq':[[...],...]} (row sums = 1)."""
    if not hasattr(clf, "get_vweights"):
        return None
    gw = clf.get_vweights()
    if "a_sq" in gw:
        return {"kind": "global", "a_sq": np.asarray(gw["a_sq"], float).tolist()}
    labels = sorted(gw.keys())
    return {
        "kind": "label",
        "labels": [int(l) for l in labels],
        "a_sq": [np.asarray(gw[l]["a_sq"], float).tolist() for l in labels],
    }


def _row_from_predictions(model, K, eta, yt_all, yp_all, status="ok", note=""):
    if status != "ok":
        return {"model": model, "K": K, "eta": eta,
                "acc": np.nan, "bal_acc": np.nan, "f1_macro": np.nan,
                "status": status, "note": note}
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    return {"model": model, "K": K, "eta": eta, **_metrics(yt, yp),
            "status": "ok", "note": ""}


def cv_single_view_cached(prepared_folds, model_cls, view_idx, tag, K, T,
                          base_seed, config_key, cache_db):
    model_name = f"single::{tag}"
    cached = load_fold_cache(cache_db, config_key)
    yt_all, yp_all = [], []

    for pf in prepared_folds:
        fold = pf["fold"]
        if fold in cached:
            c = cached[fold]
            if c["status"] != "ok":
                return _row_from_predictions(model_name, K, None, [], [], c["status"], c["note"])
            yt_all.append(c["y_true"]); yp_all.append(c["y_pred"])
            continue

        D_tr = pf["DL_tr"][view_idx]
        D_va = pf["DL_va"][view_idx]

        # mglvq_fast uses NumPy global RNG state.
        st = np.random.get_state()
        try:
            np.random.seed(base_seed + fold)
            clf = model_cls(K=K, T=T)
            clf.fit(D_tr, pf["y_tr"])
            pred = clf.predict(D_va).astype(int)
        finally:
            np.random.set_state(st)

        protos = np.asarray(clf._w).tolist() if hasattr(clf, "_w") else None
        save_fold_cache(cache_db, config_key, fold, model_name, K, None,
                        "ok", "", pf["y_va"], pred,
                        va_idx=pf["va"], vweights=None, protos=protos)
        yt_all.append(pf["y_va"]); yp_all.append(pred)

    return _row_from_predictions(model_name, K, None, yt_all, yp_all)


def cv_multiview_cached(prepared_folds, cls, name, K0, K1, eta, T,
                        base_seed, config_key, cache_db, v_init=None):
    K_text = f"{K0}/{K1}"
    cached = load_fold_cache(cache_db, config_key)
    yt_all, yp_all = [], []

    for pf in prepared_folds:
        fold = pf["fold"]
        if fold in cached:
            c = cached[fold]
            if c["status"] != "ok":
                return _row_from_predictions(name, K_text, eta, [], [], c["status"], c["note"])
            yt_all.append(c["y_true"]); yp_all.append(c["y_pred"])
            continue

        try:
            kw = dict(K={0: K0, 1: K1}, T=T, eta=eta, base_seed=base_seed + fold)
            if v_init is not None:
                # fixed-weight path: pass the start weights; eta=0 keeps them frozen.
                kw["v_init"] = np.asarray(v_init, dtype=float)
            clf = cls(**kw)
            clf.fit(pf["DL_tr"], pf["y_tr"])
            pred = clf.predict(pf["DL_va"]).astype(int)

            vweights = extract_vweights(clf)
            protos = np.asarray(clf._w).tolist() if hasattr(clf, "_w") else None

            save_fold_cache(cache_db, config_key, fold, name, K_text, eta,
                            "ok", "", pf["y_va"], pred,
                            va_idx=pf["va"], vweights=vweights, protos=protos)
            yt_all.append(pf["y_va"]); yp_all.append(pred)

        except DegenerateInitError as e:
            note = str(e).split(".")[0]
            save_fold_cache(cache_db, config_key, fold, name, K_text, eta,
                            "degenerate", note, None, None,
                            va_idx=None, vweights=None, protos=None)
            return _row_from_predictions(name, K_text, eta, [], [], "degenerate", note)

    return _row_from_predictions(name, K_text, eta, yt_all, yp_all)


# ------------------------------------------------------------------ run signature
def run_signature(T, n_splits, random_state, base_seed):
    """Fold-layout signature; the seed lives here, so a repeat is just a new rs."""
    return f"T{T}|cv{n_splits}|rs{random_state}|bs{base_seed}"


# ---------------------------------------------------------------- complete-config
def _row_from_complete_fold_cache(job, cache_db):
    cached = load_fold_cache(cache_db, job["key"])
    n_splits = job["n_splits"]

    for c in cached.values():
        if c["status"] != "ok":
            K = job.get("K", f'{job.get("K0")}/{job.get("K1")}')
            return _row_from_predictions(job["model"], K, job.get("eta"),
                                         [], [], c["status"], c["note"])
    if len(cached) != n_splits:
        return None

    yt_all = [cached[f]["y_true"] for f in range(1, n_splits + 1)]
    yp_all = [cached[f]["y_pred"] for f in range(1, n_splits + 1)]
    K = job.get("K", f'{job.get("K0")}/{job.get("K1")}')
    return _row_from_predictions(job["model"], K, job.get("eta"), yt_all, yp_all)


def _run_one_job(job, prepared_folds, cache_db, base_seed):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if job["kind"] == "single":
            row = cv_single_view_cached(
                prepared_folds=prepared_folds, model_cls=job["cls"],
                view_idx=job["view_idx"], tag=job["tag"], K=job["K"], T=job["T"],
                base_seed=base_seed, config_key=job["key"], cache_db=cache_db,
            )
        else:
            row = cv_multiview_cached(
                prepared_folds=prepared_folds, cls=job["cls"], name=job["model"],
                K0=job["K0"], K1=job["K1"], eta=job["eta"], T=job["T"],
                base_seed=base_seed, config_key=job["key"], cache_db=cache_db,
                v_init=job.get("v_init"),   # set only for fixed-weight jobs
            )
    row["key"] = job["key"]
    return job["key"], row


def run_jobs(jobs, DL, y, n_splits, cache_db, *,
             n_jobs=2, base_seed=42, random_state=42,
             legacy_results=None, stage_name="stage", progress=True):
    """Run a set of jobs at one fidelity; reuse cache and resume partial folds.

    Parameters mirror the old ``run_stage`` but are explicit (no notebook globals).
    Returns a DataFrame with one row per job (metrics + status).
    """
    from joblib import Parallel, delayed

    if not jobs:
        return pd.DataFrame()

    init_cache_db(cache_db)
    prepared_folds = prepare_folds(DL, y, n_splits, random_state)
    legacy_results = legacy_results or {}

    stage_results = {}
    pending = []
    for job in jobs:
        key = job["key"]
        if key in legacy_results:
            row = dict(legacy_results[key]); row["key"] = key
            stage_results[key] = row
            continue
        row = _row_from_complete_fold_cache(job, cache_db)
        if row is not None:
            row["key"] = key
            stage_results[key] = row
            continue
        pending.append(job)

    if pending:
        if progress:
            try:
                from tqdm.notebook import tqdm
                iterator = tqdm(total=len(jobs), initial=len(stage_results),
                                desc=stage_name, unit="cfg", dynamic_ncols=True)
            except Exception:
                iterator = None
        else:
            iterator = None

        parallel = Parallel(
            n_jobs=n_jobs, backend="loky", return_as="generator_unordered",
            batch_size=1, pre_dispatch=n_jobs, max_nbytes="10M", mmap_mode="r",
        )(delayed(_run_one_job)(job, prepared_folds, cache_db, base_seed) for job in pending)

        for key, row in parallel:
            stage_results[key] = row
            if iterator is not None:
                iterator.update(1)
                iterator.set_postfix_str(f"done: {key[:60]}", refresh=True)
        if iterator is not None:
            iterator.close()

    ordered = []
    for job in jobs:
        if job["key"] in stage_results:
            r = dict(stage_results[job["key"]]); r["stage"] = stage_name
            ordered.append(r)

    del prepared_folds
    gc.collect()
    return pd.DataFrame(ordered)
