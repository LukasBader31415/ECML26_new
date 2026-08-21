"""
convergence.py — Experiment 3: learning trajectories for the convergence-vs-
regularization question, built on the per-epoch snapshots the models already keep.

Both models snapshot loss / weights / prototypes every epoch (track_metrics /
track_vweights / track_path -> _loss, _v_history, _w_history). We fit ONCE per
(model, K0, K1, eta, fold) at a long T, then REPLAY the stored state at each
checkpoint through the model's own predict() to get validation performance at that
iteration -- no re-fit, no model hook. The raw history is written before best-so-far
restore, so the trajectory is uncontaminated by best-so-far.

Per (model, K0, K1, eta, fold, checkpoint, scope) we record:
  train_loss, w_{naics,hs,am} (the a^2 MIXTURE, sum 1), entropy (log3-normalized),
  weight_drift (||mix(t)-mix(prev cp)||2), label_distance S_t = ||mix0 - mix1||2,
  proto_drift (index-set turnover between consecutive checkpoints),
  val_bal_acc, val_recall.
scope is "global" for the global model and "label0"/"label1" for the label model
(per-class trajectories are kept separate -- central to the label story).

Output: one Parquet part file per (model,K0,K1,eta) under out_dir/parts/ (crash-safe /
resumable); load_convergence(out_dir) concatenates them.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

from .models import GlobalNew, LabelNew

VIEW_TAGS = ("naics", "hs", "am")
DEFAULT_CHECKPOINTS = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500]
DEFAULT_K_PAIRS = [(3, 3), (4, 3), (5, 3), (8, 8)]
DEFAULT_ETAS = [0.005, 0.01]

_MODELS = {"M3GLVQ_Global": GlobalNew, "M3GLVQ_Label": LabelNew}


def _entropy(mix):
    """log3-normalized entropy of a length-3 mixture (sum 1). 1=uniform, 0=one view."""
    m = np.clip(np.asarray(mix, float), 0, None)
    s = m.sum()
    if s <= 0:
        return np.nan
    m = m / s
    nz = m[m > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(3))


def _mixtures(v_state, kind):
    """Return {scope: mix(a^2)} from a stored weight snapshot.
    global: v_state is (V,) a-vector      -> {'global': a^2}
    label : v_state is (L,V) a-matrix     -> {'label0': a0^2, 'label1': a1^2}"""
    v = np.asarray(v_state, float)
    if kind == "global":
        return {"global": v ** 2}
    return {f"label{l}": v[l] ** 2 for l in range(v.shape[0])}


def _proto_turnover(w_now, w_prev):
    if w_prev is None:
        return np.nan
    a, b = set(map(int, w_now)), set(map(int, w_prev))
    return float(1.0 - len(a & b) / max(1, len(a | b)))


def _clamp_cp(cp, n_hist):
    """Map a requested checkpoint iteration to an available history index."""
    return min(int(cp), n_hist - 1)


def _replay_predict(clf, kind, w_state, v_state, DL_va):
    """Set the model to a checkpoint state and predict on the validation fold."""
    clf._w = np.asarray(w_state, dtype=int)
    if kind == "global":
        clf._vWeights = np.asarray(v_state, float)
    else:
        clf._vWeights_ls = np.asarray(v_state, float)
    return clf.predict(DL_va).astype(int)


def _fit_one(model_name, cls, DL, y, tr, va, K0, K1, eta, T, base_seed):
    DL_tr = [D[np.ix_(tr, tr)] for D in DL]
    DL_va = [D[np.ix_(va, tr)] for D in DL]
    clf = cls(K={0: K0, 1: K1}, T=T, eta=eta, base_seed=base_seed,
              track_path=True, track_vweights=True, track_metrics=True)
    clf.fit(DL_tr, y[tr])
    return clf, DL_va


def run_convergence(DL, y, out_dir, *,
                    k_pairs=DEFAULT_K_PAIRS, etas=DEFAULT_ETAS,
                    models=("M3GLVQ_Global", "M3GLVQ_Label"),
                    T=500, checkpoints=DEFAULT_CHECKPOINTS,
                    n_splits=10, random_state=42, base_seed=42,
                    progress=True, resume=True):
    """Run the trajectory experiment; write one Parquet part per (model,K0,K1,eta).
    Folds are identical across all configs (same random_state) -> paired ΔBA later."""
    out_dir = Path(out_dir); parts = out_dir / "parts"; parts.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(np.zeros(len(y)), y))

    total = len(models) * len(k_pairs) * len(etas)
    done = 0
    for model_name in models:
        cls = _MODELS[model_name]
        kind = "global" if model_name.endswith("Global") else "label"
        for (K0, K1) in k_pairs:
            for eta in etas:
                part = parts / f"{model_name}_K{K0}-{K1}_eta{eta}.parquet"
                done += 1
                if resume and part.exists():
                    if progress:
                        print(f"[conv] skip cached {part.name} ({done}/{total})", flush=True)
                    continue

                rows = []
                for fold, (tr, va) in enumerate(folds, start=1):
                    try:
                        clf, DL_va = _fit_one(model_name, cls, DL, y, tr, va,
                                              K0, K1, eta, T, base_seed + fold)
                    except Exception as e:
                        rows.append(dict(model=model_name, K0=K0, K1=K1, eta=eta, fold=fold,
                                         checkpoint=-1, status=f"failed:{str(e)[:60]}"))
                        continue

                    n_hist = len(clf._v_history)
                    n_epochs_run = n_hist - 1          # snapshots = 1 (init) + epochs run
                    # loss history aligned with _v_history (both filled in the same
                    # _snapshot call); _log["loss"] exists because track_metrics=True.
                    loss_hist = (clf._log["loss"] if getattr(clf, "_log", None)
                                 and "loss" in clf._log else clf._loss)
                    y_va = y[va]
                    prev_mix = {}; prev_w = None
                    for cp in checkpoints:
                        idx = _clamp_cp(cp, n_hist)
                        v_state = clf._v_history[idx]
                        w_state = clf._w_history[idx]
                        train_loss = float(loss_hist[idx]) if idx < len(loss_hist) else np.nan

                        pred = _replay_predict(clf, kind, w_state, v_state, DL_va)
                        val_ba = balanced_accuracy_score(y_va, pred)
                        val_rec = recall_score(y_va, pred, pos_label=1, zero_division=0)
                        proto_dr = _proto_turnover(w_state, prev_w)

                        mixes = _mixtures(v_state, kind)
                        S_t = np.nan
                        if kind == "label" and "label0" in mixes and "label1" in mixes:
                            S_t = float(np.linalg.norm(mixes["label0"] - mixes["label1"]))

                        for scope, mix in mixes.items():
                            drift = (np.nan if scope not in prev_mix
                                     else float(np.linalg.norm(mix - prev_mix[scope])))
                            rows.append(dict(
                                model=model_name, K0=K0, K1=K1, eta=eta, fold=fold,
                                checkpoint=int(cp), iter_used=int(idx), status="ok",
                                n_epochs_run=int(n_epochs_run),
                                converged_before_T=bool(n_epochs_run < T),
                                scope=scope, train_loss=train_loss,
                                w_naics=float(mix[0]), w_hs=float(mix[1]), w_am=float(mix[2]),
                                entropy=_entropy(mix), weight_drift=drift,
                                label_distance=S_t, proto_drift=proto_dr,
                                val_bal_acc=float(val_ba), val_recall=float(val_rec),
                            ))
                            prev_mix[scope] = mix
                        prev_w = w_state

                pd.DataFrame(rows).to_parquet(part, index=False)
                if progress:
                    print(f"[conv] wrote {part.name} ({done}/{total})", flush=True)

    return load_convergence(out_dir)


def load_convergence(out_dir):
    parts = sorted((Path(out_dir) / "parts").glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def paired_delta_ba(traj, cp_a, cp_b, scope="global"):
    """Paired ΔBA per fold between two checkpoints (same folds) + mean/SD.
    Returns (per_fold_df, summary_df)."""
    d = traj[(traj.scope == scope) & (traj.status == "ok") &
             (traj.checkpoint.isin([cp_a, cp_b]))]
    piv = d.pivot_table(index=["model", "K0", "K1", "eta", "fold"],
                        columns="checkpoint", values="val_bal_acc")
    piv = piv.dropna(subset=[cp_a, cp_b])
    piv["delta_ba"] = piv[cp_b] - piv[cp_a]
    per_fold = piv.reset_index()
    summ = (per_fold.groupby(["model", "K0", "K1", "eta"])["delta_ba"]
                     .agg(mean_delta="mean", sd_delta="std", n="count").reset_index())
    return per_fold, summ