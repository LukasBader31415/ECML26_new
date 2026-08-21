"""
weight_tail.py — Experiment A1: weight-only continuation (diagnosis, NO model change).

The models stop when the discrete prototype search finds no improving swap; because
the weight update is nested inside the "a swap happened" branch, prototype
convergence also freezes the view weights. A1 isolates the weight question:

  1. fit exactly as usual  -> prototypes frozen at the natural stop, weights ~uniform
  2. FREEZE prototypes (clf._w fixed) and keep ONLY the view-weight update running for
     n_extra epochs, re-assigning points under the current D* each epoch
  3. record train_loss / val_bal_acc / entropy / label_distance / weight_drift per cp

Three outcomes (per the plan):
  - weights don't move            -> the early stop was harmless for the weights too
  - weights move, loss down, BA up -> the stop cut weight-learning short (fix worth it)
  - weights specialize, BA down    -> further training specializes but hurts generalization

This driver calls only real model methods (_overall / _weights_update[_label]) and
mirrors the in-fit assignment closures. A bit-check asserts the tail's epoch-0 loss
reproduces the fitted best-so-far loss, so the mirrored assignment is provably faithful.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

from .models import GlobalNew, LabelNew
from .convergence import _entropy, _mixtures

_MODELS = {"M3GLVQ_Global": GlobalNew, "M3GLVQ_Label": LabelNew}
_BITCHECK_TOL = 1e-6


def _phi(clf, x):
    return clf.phi(x) if hasattr(clf, "phi") else x


# ---------------------------------------------------------------- assignments
def _assign_global(clf, M, overall, y_tr):
    """Mirror of the in-fit _assign_all for the global model (prototypes clf._w)."""
    m = clf._m; V = clf._V
    rows = np.arange(m)
    w = clf._w
    labs = np.unique(y_tr)
    cp = np.zeros(m, dtype=int); cm = np.zeros(m, dtype=int)
    for lab in labs:
        pts = np.where(y_tr == lab)[0]
        iw = np.where(clf._y == lab)[0]
        ow = np.where(clf._y != lab)[0]
        cp[pts] = iw[np.argmin(overall[np.ix_(pts, w[iw])], axis=1)]
        cm[pts] = ow[np.argmin(overall[np.ix_(pts, w[ow])], axis=1)]
    dp = overall[rows, w[cp]]; dm = overall[rows, w[cm]]
    dp_V = M[:, rows, w[cp]]; dm_V = M[:, rows, w[cm]]
    return dp, dm, dp_V, dm_V


def _assign_label(clf, M, overall_L, y_tr):
    """Mirror of the in-fit _assign for the label model (per-label metrics)."""
    m = clf._m; V = clf._V; w = clf._w
    labs = clf._ls_labels
    cp = np.zeros(m, dtype=int); cm = np.zeros(m, dtype=int)
    dp = np.zeros(m); dm = np.zeros(m)
    dp_V = np.zeros((V, m)); dm_V = np.zeros((V, m))
    for l, lab in enumerate(labs):
        inClass = np.where(y_tr == lab)[0]
        ov = overall_L[l]
        in_w = np.where(clf._y == lab)[0]
        out_w = np.where(clf._y != lab)[0]
        cp[inClass] = in_w[np.argmin(ov[inClass][:, w[in_w]], axis=1)]
        cm[inClass] = out_w[np.argmin(ov[inClass][:, w[out_w]], axis=1)]
        dp[inClass] = ov[inClass, w[cp[inClass]]]
        dm[inClass] = ov[inClass, w[cm[inClass]]]
        dp_V[:, inClass] = M[:, inClass, w[cp[inClass]]]
        dm_V[:, inClass] = M[:, inClass, w[cm[inClass]]]
    return dp, dm, dp_V, dm_V


def _loss(clf, dp, dm):
    return float(_phi(clf, (dp - dm) / (dp + dm + 1e-5)).sum())


# ---------------------------------------------------------------- tail driver
def weight_only_tail(clf, DL_tr, y_tr, kind, n_extra, checkpoints,
                     DL_va=None, y_va=None):
    """Run n_extra weight-only epochs with prototypes frozen; return a list of rows."""
    M = np.stack(DL_tr, axis=0)
    y_tr = np.asarray(y_tr, dtype=int)

    # --- bit-check: epoch-0 loss must reproduce the fitted best-so-far loss ---
    if kind == "global":
        ov = clf._overall(M)
        dp, dm, dp_V, dm_V = _assign_global(clf, M, ov, y_tr)
    else:
        ovL = [clf._overall_for_label(M, l) for l in range(len(clf._ls_labels))]
        dp, dm, dp_V, dm_V = _assign_label(clf, M, ovL, y_tr)
    loss0 = _loss(clf, dp, dm)
    ref = float(min(clf._loss))                       # best-so-far loss
    if abs(loss0 - ref) > _BITCHECK_TOL * max(1.0, abs(ref)):
        raise RuntimeError(
            f"A1 assignment bit-check failed: tail loss0={loss0:.6f} != fitted "
            f"best={ref:.6f} (mirrored assignment diverges from the model).")

    rows = []
    prev_mix = {}
    checkpoints = sorted(set([0] + list(checkpoints)))

    def _record(ep, dp, dm):
        v_state = clf._vWeights if kind == "global" else clf._vWeights_ls
        mixes = _mixtures(v_state, kind)
        train_loss = _loss(clf, dp, dm)
        val_ba = val_rec = np.nan
        if DL_va is not None:
            pred = clf.predict(DL_va).astype(int)
            val_ba = balanced_accuracy_score(y_va, pred)
            val_rec = recall_score(y_va, pred, pos_label=1, zero_division=0)
        S_t = (np.nan if kind == "global"
               else float(np.linalg.norm(mixes["label0"] - mixes["label1"])))
        for scope, mix in mixes.items():
            drift = (np.nan if scope not in prev_mix
                     else float(np.linalg.norm(mix - prev_mix[scope])))
            rows.append(dict(tail_epoch=int(ep), scope=scope, train_loss=train_loss,
                             w_naics=float(mix[0]), w_hs=float(mix[1]), w_am=float(mix[2]),
                             entropy=_entropy(mix), weight_drift=drift,
                             label_distance=S_t, val_bal_acc=float(val_ba),
                             val_recall=float(val_rec)))
            prev_mix[scope] = mix

    _record(0, dp, dm)                                # state at the natural stop

    for ep in range(1, n_extra + 1):
        if kind == "global":
            ov = clf._overall(M)
            dp, dm, dp_V, dm_V = _assign_global(clf, M, ov, y_tr)
            clf._weights_update(dp, dm, dp_V, dm_V)   # protos untouched
        else:
            ovL = [clf._overall_for_label(M, l) for l in range(len(clf._ls_labels))]
            dp, dm, dp_V, dm_V = _assign_label(clf, M, ovL, y_tr)
            for l, lab in enumerate(clf._ls_labels):
                mask = (y_tr == lab)
                clf._weights_update_label(l, dp, dm, dp_V, dm_V, mask)
        if ep in checkpoints:
            # re-assign under the updated weights for an accurate post-step loss
            if kind == "global":
                dp, dm, dp_V, dm_V = _assign_global(clf, M, clf._overall(M), y_tr)
            else:
                ovL = [clf._overall_for_label(M, l) for l in range(len(clf._ls_labels))]
                dp, dm, dp_V, dm_V = _assign_label(clf, M, ovL, y_tr)
            _record(ep, dp, dm)
    return rows


def run_weight_tail(DL, y, out_dir, *,
                    k_pairs=((3, 3), (4, 3), (5, 3)), etas=(0.005, 0.01),
                    models=("M3GLVQ_Global", "M3GLVQ_Label"),
                    T_fit=500, n_extra=200,
                    checkpoints=(0, 10, 25, 50, 75, 100, 150, 200),
                    n_splits=10, random_state=42, base_seed=42,
                    progress=True, resume=True):
    """Fit normally, then run a weight-only tail; one Parquet part per (model,K,eta)."""
    out_dir = Path(out_dir); parts = out_dir / "parts"; parts.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(np.zeros(len(y)), y))

    import time as _time
    total_fits = len(models) * len(k_pairs) * len(etas) * n_splits
    fits_done = 0; t0 = _time.time()

    def _tick(tag):
        if not progress:
            return
        el = _time.time() - t0
        rate = fits_done / el if el > 0 else 0.0
        eta = (total_fits - fits_done) / rate if rate > 0 else float("nan")
        print(f"\r[A1] {fits_done}/{total_fits} fits | {rate:5.2f} fit/s | "
              f"elapsed {el:6.0f}s | ETA {eta:6.0f}s | {tag[:32]}   ", end="", flush=True)

    total = len(models) * len(k_pairs) * len(etas); done = 0
    for model_name in models:
        cls = _MODELS[model_name]; kind = "global" if model_name.endswith("Global") else "label"
        for (K0, K1) in k_pairs:
            for eta in etas:
                part = parts / f"{model_name}_K{K0}-{K1}_eta{eta}.parquet"; done += 1
                if resume and part.exists():
                    fits_done += n_splits
                    if progress: print(f"\r[A1] skip {part.name} ({done}/{total})            ", flush=True)
                    continue
                rows = []
                for fold, (tr, va) in enumerate(folds, start=1):
                    DL_tr = [D[np.ix_(tr, tr)] for D in DL]
                    DL_va = [D[np.ix_(va, tr)] for D in DL]
                    clf = cls(K={0: K0, 1: K1}, T=T_fit, eta=eta, base_seed=base_seed + fold,
                              track_vweights=True, track_metrics=True)
                    try:
                        clf.fit(DL_tr, y[tr])
                        stop_epoch = len(clf._v_history) - 1
                        tail = weight_only_tail(clf, DL_tr, y[tr], kind, n_extra,
                                                checkpoints, DL_va=DL_va, y_va=y[va])
                    except Exception as e:
                        rows.append(dict(model=model_name, K0=K0, K1=K1, eta=eta,
                                         fold=fold, tail_epoch=-1, status=f"failed:{str(e)[:70]}"))
                        fits_done += 1; _tick(f"{model_name} K{K0}/{K1} eta{eta} f{fold} FAIL")
                        continue
                    for r in tail:
                        r.update(model=model_name, K0=K0, K1=K1, eta=eta, fold=fold,
                                 stop_epoch=int(stop_epoch), status="ok")
                        rows.append(r)
                    fits_done += 1; _tick(f"{model_name} K{K0}/{K1} eta{eta} f{fold}")
                pd.DataFrame(rows).to_parquet(part, index=False)
                if progress: print(f"\n[A1] wrote {part.name} ({done}/{total})", flush=True)

    return load_weight_tail(out_dir)


def load_weight_tail(out_dir):
    parts = sorted((Path(out_dir) / "parts").glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)