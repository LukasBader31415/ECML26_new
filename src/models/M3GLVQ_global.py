__author__ = 'Lukas Bader, Dietlind Zühlke'
__copyright__ = ('Copyright 2019, Benjamin Paaßen; '
                 'Copyright 2026, Lukas Bader, Dietlind Zühlke')
__license__ = 'GPLv3'
__version__ = '2.1.0-global'
__maintainer__ = 'Lukas Bader'
__email__ = 'lukas.bader@pferd.com'


"""
M3GLVQ -- GLOBAL variant (one shared squared-weight vector)

This module implements the M3GLVQ algorithm as described in:

    Lukas Bader, Ina Terwey-Scheulen, Dietlind Zühlke,
    "Implementation of Multi-Matrix Median Generalized Learning Vector Quantization",
    ESANN 2026.

The implementation is based on and extends the original MGLVQ code by
Benjamin Paaßen (proto-dist-ml, GNU GPLv3).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from src.common.proto_init import init_class_prototypes
_ERR_CUTOFF = 1e-5

def _project_simplex(v):
    """Project vector v onto the probability simplex {w >= 0, sum(w)=1}. (Duchi et al., 2008).

    Kept for backward compatibility / non-global paths. The global path now uses
    the squared-weight parametrization and _normalize_l2 instead.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector.")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def _normalize_l2(a):
    """Normalize vector a onto the unit L2-sphere {sum(a_v^2) = 1}.

    Squared-weight parametrization: the distance uses a_v^2 as group weights, so
    sum(a_v^2)=1  <=>  ||a||_2 = 1. Non-negativity of the effective weights a_v^2
    is automatic regardless of the sign of a_v, hence no simplex projection is
    needed in the global path.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("L2 normalization expects a 1D vector.")
    nrm = np.linalg.norm(a, 2)
    if nrm < 1e-12:
        # degenerate input -> fall back to uniform a_v (so a_v^2 = 1/V each)
        return np.full(a.size, 1.0 / np.sqrt(a.size))
    return a / nrm


class M3GLVQ_Global(BaseEstimator, ClassifierMixin):
    """
    VMGLVQ: Median Generalized LVQ for multiple distance matrices (DL = [D^(v)]).
    - Combines distances: D* = Σ_v a_v^2 D^(v) with learnable factors a_v.
      Squared-weight parametrization (global path): the effective group weight is
      a_v^2, which is non-negative by construction, so NO simplex projection is
      needed. The only constraint is Σ_v a_v^2 = 1, i.e. ||a||_2 = 1, enforced by
      simple L2 normalization after each gradient step.
    - get_vweight_path() / final weights expose BOTH a_v and a_v^2.
    - Vectorized implementation (no O(V·m²) double loops).
    - Supports K>1 as well as K==1 (multi-class/binary) with specialized paths.
    - Optional tracking: prototype path, weights, loss history.

    Parameters
    ----------
    K : int | dict[label -> int] | sequence[int]
        Number of prototypes per label (int = uniform for all labels).
    T : int
        Epochs.
    phi : callable or None
        Squashing function applied to the GLVQ µ-values (default: identity).
    track_path : bool
    track_vweights : bool
    track_metrics : bool
    eta : float
        Learning rate for matrix weights.
    v_init : array-like or None
        Initial factors a_v of the V matrices; L2-normalized so that Σ a_v^2 = 1.
    """

    def __init__(
        self,
        K,
        T=50,
        phi=None,
        *,
        track_path=False,
        track_vweights=True,
        track_metrics=False,
        eta=1.0,
        v_init=None,
        base_seed=42,
    ):
        self.K = K
        self.T = T
        self.phi = (lambda mus: mus) if phi is None else phi

        self.track_path = bool(track_path)
        self.track_vweights = bool(track_vweights)
        self.track_metrics = bool(track_metrics)
        self.eta = float(eta)
        self.v_init = None if v_init is None else np.asarray(v_init, dtype=float)
        self.base_seed = int(base_seed)

        self._w_history = None
        self._v_history = None
        self._log = None
        self._snapshot_idx = 0

        self.final_matrix_ = None  # Set after fit()

    # ---------- Tracking ----------
    def _init_tracking(self):
        self._snapshot_idx = 0
        self._w_history = [] if self.track_path else None
        self._v_history = [] if self.track_vweights else None
        self._log = {"loss": []} if self.track_metrics else None

    def _snapshot(self, *, w=None, v=None, loss=None):
        if self.track_path and w is not None:
            self._w_history.append(np.array(w, copy=True))
        if self.track_vweights and v is not None:
            self._v_history.append(np.array(v, dtype=float, copy=True))
        if self.track_metrics and loss is not None:
            self._log["loss"].append(float(loss))
        self._snapshot_idx += 1

    def get_final_matrix(self):
        if self.final_matrix_ is None:
            raise RuntimeError("Model has not been trained yet (fit has not been called).")
        return self.final_matrix_

    # ---------- Utilities ----------
    def _resolve_K_per_label(self, unique_labels):
        """
        self.K may be:
          - int → same number per label
          - dict {label: k_l}
          - sequence[int] in the order of unique_labels
        Returns: K_per (L,) as int array.
        """
        L = len(unique_labels)
        if isinstance(self.K, (int, np.integer)):
            if self.K < 1:
                raise ValueError("K must be >= 1")
            return np.full(L, int(self.K), dtype=int)

        if isinstance(self.K, dict):
            K_per = np.empty(L, dtype=int)
            for i, lab in enumerate(unique_labels):
                if lab not in self.K:
                    raise ValueError(f"K missing for label {lab!r}")
                K_per[i] = int(self.K[lab])
            if np.any(K_per < 1):
                raise ValueError("All K_l must be >= 1")
            return K_per

        try:
            K_seq = np.asarray(self.K, dtype=int)
            if K_seq.shape != (L,):
                raise ValueError(f"K must have length {L}; got shape {K_seq.shape}")
            if np.any(K_seq < 1):
                raise ValueError("All K_l must be >= 1")
            return K_seq
        except Exception as e:
            raise ValueError("Unsupported K format. Use int, dict {label:k}, or sequence length=L") from e

    def _stack_DL(self, DL):
        """DL → M with shape (V, m, m) and set base dimensions; respect v_init."""
        M = np.stack(DL, axis=0)
        self._V = M.shape[0]
        self._m = M.shape[1]

        if self.v_init is not None:
            if self.v_init.shape != (self._V,):
                raise ValueError(f"v_init must have shape ({self._V},), got {self.v_init.shape}")
            # squared-weight parametrization: L2-normalize so that Σ a_v^2 = 1
            self._vWeights = _normalize_l2(self.v_init)
        elif not hasattr(self, "_vWeights") or self._vWeights is None or self._vWeights.size != self._V:
            # uniform start: a_v = 1/sqrt(V)  ->  a_v^2 = 1/V,  Σ a_v^2 = 1
            self._vWeights = np.full(self._V, 1.0 / np.sqrt(self._V))

        if not hasattr(self, "_v_m") or self._v_m is None or self._v_m.size != self._V:
            self._v_m = np.zeros(self._V, dtype=float)

        return M

    def _overall(self, M):
        """Squared-weighted sum over V -> shape (m, m): D* = Σ_v a_v^2 D^(v)."""
        return np.tensordot(self._vWeights ** 2, M, axes=(0, 0))

    def _weights_update(self, dp, dm, dp_V, dm_V):
        """
        Gradient step for the matrix factors a_v under the SQUARED-weight
        parametrization D* = Σ_v a_v^2 D^(v), followed by L2 normalization.

        Per data point i the contribution to dCF/da_v is (chain rule, see ESANN
        derivation):

            dCF/da_v = Σ_i phi'_i * 4 * a_v * (d_iv^+ * d_i^-  -  d_iv^- * d_i^+)
                                              / (d_i^+ + d_i^-)^2

        Here phi'_i is set to 1 (identity squashing); dp, dm are the combined
        d_i^+, d_i^-; dp_V[v,i], dm_V[v,i] are the per-matrix d_iv^+, d_iv^-.
        We descend the loss, hence the negative gradient is used as step.

        dp, dm     : (m,)      combined d^+ , d^-
        dp_V, dm_V : (V, m)    per-matrix d_iv^+ , d_iv^-
        """
        den = (dp + dm + 1e-5) ** 2                      # (m,)
        # num_v,i = d_iv^+ * d_i^-  -  d_iv^- * d_i^+
        num = dp_V * dm[None, :] - dm_V * dp[None, :]     # (V, m)
        # dCF/da_v  (factor 4 * a_v from the squared parametrization)
        grad = (4.0 * self._vWeights[:, None] * num / den[None, :]).sum(axis=1)

        # normalize the gradient direction for a stable, scale-free step
        g_norm = np.linalg.norm(grad, 2) + 1e-12
        step_vec = grad / g_norm

        # gradient DESCENT on the loss -> move against the gradient
        a_new = self._vWeights - self.eta * step_vec
        # only constraint: Σ a_v^2 = 1  ->  L2 normalization (no simplex needed)
        self._vWeights = _normalize_l2(a_new)

    # ---------- Fit: general path (arbitrary K_l) ----------
    def fit(self, DL, y):
        self._init_tracking()

        # Stack
        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)

        K_per = self._resolve_K_per_label(unique_labels)
        total_K = int(K_per.sum())

        self._y = np.repeat(unique_labels, K_per)
        self._w = np.zeros(total_K, dtype=int)

        if np.all(K_per == 1):
            return self._fit_single(DL, y)

        # prototype initialization: uniform hard-RNG (Relational Neural Gas) via
        # the shared proto_init helper. Seeded per (base_seed, label) for
        # reproducibility; raises on degenerate small classes (no medoid fallback).
        if (not hasattr(self, 'prevent_initialization')) or (not self.prevent_initialization):
            overall = self._overall(M)
            offset = 0
            for l, lab in enumerate(unique_labels):
                k_l = int(K_per[l])
                idx_w = np.arange(offset, offset + k_l)
                inClass = np.where(y == lab)[0]
                self._w[idx_w] = init_class_prototypes(
                    overall, inClass, k_l,
                    base_seed=self.base_seed, label=lab, view_tag="global",
                )
                offset += k_l

        rows = np.arange(self._m)
        # fixed groupings (independent of the candidate being tried)
        pts_by_l = [np.where(y == lab)[0] for lab in unique_labels]
        in_w_by_l = [np.where(self._y == lab)[0] for lab in unique_labels]
        out_w_by_l = [np.where(self._y != lab)[0] for lab in unique_labels]
        lab_to_idx = {lab: l for l, lab in enumerate(unique_labels)}

        def _assign_all(w, ov):
            """Nearest own/wrong prototype per point under the combined metric
            ov; returns closest_plus/minus and combined + per-matrix d^+, d^-."""
            cp = np.zeros(self._m, dtype=int)
            cm = np.zeros(self._m, dtype=int)
            for l in range(len(unique_labels)):
                pts = pts_by_l[l]; iw = in_w_by_l[l]; ow = out_w_by_l[l]
                cp[pts] = iw[np.argmin(ov[np.ix_(pts, w[iw])], axis=1)]
                cm[pts] = ow[np.argmin(ov[np.ix_(pts, w[ow])], axis=1)]
            dp_ = ov[rows, w[cp]]; dm_ = ov[rows, w[cm]]
            dp_V_ = M[:, rows, w[cp]]; dm_V_ = M[:, rows, w[cm]]
            return cp, cm, dp_, dm_, dp_V_, dm_V_

        def _dp_dm(w, ov):   # exact reference (used by >2-class fallback + harness)
            dp_ = np.empty(self._m); dm_ = np.empty(self._m)
            for l in range(len(unique_labels)):
                pts = pts_by_l[l]
                dp_[pts] = ov[np.ix_(pts, w[in_w_by_l[l]])].min(axis=1)
                dm_[pts] = ov[np.ix_(pts, w[out_w_by_l[l]])].min(axis=1)
            return dp_, dm_

        def _loss(dp_, dm_):
            return float(self.phi((dp_ - dm_) / (dp_ + dm_ + 1e-5)).sum())

        def _scan_k(k, ov, dp, dm, closest_plus, cur_loss):
            """Best (i, delta) for moving prototype k, all rf_plus candidates at
            once. Binary case is vectorized and BIT-IDENTICAL to the per-candidate
            loop (identical min values; loss summed in index order via
            mu.sum(axis=0), matching _loss). >2 classes fall back to the loop."""
            rf_plus = np.where(closest_plus == k)[0]
            cand = rf_plus[rf_plus != self._w[k]]
            if cand.size == 0:
                return None, 0.0

            if len(unique_labels) != 2:
                best_delta = 0.0; best_i = None
                w_try = self._w.copy()
                for i in cand:
                    w_try[k] = i
                    d = _loss(*_dp_dm(w_try, ov)) - cur_loss
                    if d < best_delta:
                        best_delta = d; best_i = i
                w_try[k] = self._w[k]
                return best_i, best_delta

            # --- binary vectorized: moving k (class lab_k) changes own-class d^+
            # and the other class's d^- (both over the lab_k prototype set) ---
            l_k = lab_to_idx[self._y[k]]
            own = pts_by_l[l_k]
            other = pts_by_l[1 - l_k]
            own_protos = in_w_by_l[l_k]
            protos_g = self._w[own_protos]
            kpos = int(np.where(own_protos == k)[0][0])

            Down = ov[np.ix_(own, protos_g)].copy(); Down[:, kpos] = np.inf
            dp_wo = Down.min(axis=1)                     # own d^+ excluding k
            Doth = ov[np.ix_(other, protos_g)].copy(); Doth[:, kpos] = np.inf
            dm_wo = Doth.min(axis=1)                     # other d^- excluding k

            dp_cand = np.minimum(dp_wo[:, None], ov[np.ix_(own, cand)])
            dm_cand = np.minimum(dm_wo[:, None], ov[np.ix_(other, cand)])

            mu = np.empty((self._m, cand.size))
            mu[own] = self.phi((dp_cand - dm[own][:, None]) / (dp_cand + dm[own][:, None] + 1e-5))
            mu[other] = self.phi((dp[other][:, None] - dm_cand) / (dp[other][:, None] + dm_cand + 1e-5))
            # sum per candidate in point-index order -> BIT-IDENTICAL to _loss's
            # 1D .sum() (a plain mu.sum(axis=0) differs by ~1e-12 and would create
            # spurious sub-_ERR_CUTOFF "improvements" that trip the early stop).
            deltas = np.ascontiguousarray(mu.T).sum(axis=1) - cur_loss

            j = int(np.argmin(deltas))
            if deltas[j] < 0.0:
                return int(cand[j]), float(deltas[j])
            return None, 0.0

        overall = self._overall(M)
        closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign_all(self._w, overall)
        self._loss = [_loss(dp, dm)]
        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        # best-so-far: fixed-eta weight steps can overshoot at large eta; keep the
        # lowest-loss state and return it instead of the final iterate (no-op when
        # monotone).
        best_loss = self._loss[-1]
        best_w = self._w.copy()
        best_vw = self._vWeights.copy()

        def _track_best():
            nonlocal best_loss, best_w, best_vw
            if self._loss[-1] < best_loss:
                best_loss = self._loss[-1]
                best_w = self._w.copy()
                best_vw = self._vWeights.copy()

        # `overall = sum a_v^2 D_v` depends only on the weights, not on prototype
        # positions -> rebuilt only after a weight step (carried over otherwise).
        # Candidate scan is vectorized but scores the identical objective.
        proto_losses = np.zeros(len(self._w))
        for _ in range(self.T):
            cur_loss = self._loss[-1]

            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                best_i, best_delta = _scan_k(k, overall, dp, dm, closest_plus, cur_loss)
                if best_i is None:
                    continue

                self._w[k] = best_i
                improved = True
                best_delta_global = best_delta

                # swap leaves `overall` unchanged: refresh assignments only
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign_all(self._w, overall)
                self._loss.append(_loss(dp, dm))
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                _track_best()

                # weight step changes weights -> rebuild `overall`
                self._weights_update(dp, dm, dp_V, dm_V)
                overall = self._overall(M)
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign_all(self._w, overall)
                self._loss.append(_loss(dp, dm))
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                _track_best()
                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        # return the best-so-far state (identical to final when monotone)
        self._w = best_w
        self._vWeights = best_vw
        self.final_matrix_ = self._overall(M)

        return self

    # ---------- Fit: K == 1 (multi-class) ----------
    def _fit_single(self, DL, y):
        self._init_tracking()

        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)

        self._y = np.array(unique_labels, copy=True)
        self._w = np.zeros(L, dtype=int)

        overall = self._overall(M)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            self._w[l] = int(init_class_prototypes(
                overall, inClass, 1,
                base_seed=self.base_seed, label=lab, view_tag="global",
            )[0])

        closest_plus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            closest_plus[np.where(y == lab)[0]] = l

        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            out_w = np.where(self._y != lab)[0]
            Dm = overall[inClass, :][:, self._w[out_w]]
            idx = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass] = out_w[idx[:, 0]]
            sndclosest_minus[inClass] = out_w[idx[:, 1 if len(out_w) > 1 else 0]]

        rows = np.arange(self._m)
        dp = overall[rows, self._w[closest_plus]]
        dm = overall[rows, self._w[closest_minus]]
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]

        dp_V = M[:, rows, self._w[closest_plus]]
        dm_V = M[:, rows, self._w[closest_minus]]

        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        proto_losses = np.zeros(len(self._w))
        for _ in range(self.T):
            overall = self._overall(M)

            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                inClass_k = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                rf_plus = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta = 0.0
                best = None
                for i in rf_plus:
                    if i == self._w[k]:
                        continue

                    still_m = overall[rf_minus, i] <= overall[rf_minus, self._w[sndclosest_minus[rf_minus]]]
                    changed_minus = np.unique(np.concatenate([outClass_k[overall[outClass_k, i] < dm[outClass_k]],
                                                               rf_minus[still_m]]))
                    changed_minus2 = rf_minus[~still_m]

                    delta = 0.0
                    dp_new = overall[rf_plus, i]
                    mus_new = self.phi((dp_new - dm[rf_plus]) / (dp_new + dm[rf_plus] + 1e-5))
                    delta += np.sum(mus_new - mus[rf_plus])

                    dm_new = overall[changed_minus, i]
                    mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[changed_minus])

                    dm_new = overall[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
                    mus_new = self.phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[changed_minus2])

                    if delta < best_delta:
                        best_delta = delta
                        best = (i, changed_minus, changed_minus2)

                if best is None:
                    continue

                i_best, c_m, c_m2 = best
                self._w[k] = i_best
                improved = True
                best_delta_global = best_delta

                dp[rf_plus] = overall[rf_plus, i_best]
                mus[rf_plus] = self.phi((dp[rf_plus] - dm[rf_plus]) / (dp[rf_plus] + dm[rf_plus] + 1e-5))
                dp_V[:, rf_plus] = M[:, rf_plus, i_best]

                closest_minus[c_m] = k
                dm[c_m] = overall[c_m, i_best]
                mus[c_m] = self.phi((dp[c_m] - dm[c_m]) / (dp[c_m] + dm[c_m] + 1e-5))
                dm_V[:, c_m] = M[:, c_m, i_best]

                for l, lab in enumerate(unique_labels):
                    ic = c_m[np.where(y[c_m] == lab)[0]]
                    if ic.size == 0:
                        continue
                    out_w = np.where(self._y != lab)[0]
                    idx = np.argpartition(overall[ic, :][:, self._w[out_w]], 1, axis=1)
                    if len(out_w) > 1:
                        sndclosest_minus[ic] = out_w[idx[:, 1]]
                    else:
                        sndclosest_minus[ic] = out_w[0]

                closest_minus[c_m2] = sndclosest_minus[c_m2]
                dm[c_m2] = overall[c_m2, self._w[closest_minus[c_m2]]]
                mus[c_m2] = self.phi((dp[c_m2] - dm[c_m2]) / (dp[c_m2] + dm[c_m2] + 1e-5))
                dm_V[:, c_m2] = M[:, c_m2, self._w[closest_minus[c_m2]]]
                out_w = np.where(self._y != self._y[k])[0]
                idx = np.argpartition(overall[c_m2, :][:, self._w[out_w]], 1, axis=1)
                if len(out_w) > 1:
                    sndclosest_minus[c_m2] = out_w[idx[:, 1]]

                for l, lab in enumerate(unique_labels):
                    ic = c_m2[np.where(y[c_m2] == lab)[0]]
                    if ic.size == 0:
                        continue
                    out_w = np.where(self._y != lab)[0]
                    idx = np.argpartition(overall[ic, :][:, self._w[out_w]], 1, axis=1)
                    if len(out_w) > 1:
                        sndclosest_minus[ic] = out_w[idx[:, 1]]
                    else:
                        sndclosest_minus[ic] = out_w[0]

                expected_new = self._loss[-1] + best_delta
                actual_new = float(mus.sum())
                rel_err = abs(expected_new - actual_new) / (abs(self._loss[-1]) + 1e-12)
                if rel_err > 0.05:
                    print(f"[Warning] Loss deviation: expected {expected_new:.3f}, actual {actual_new:.3f} (rel_err={rel_err:.3%})")

                self._loss.append(actual_new)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                self._weights_update(dp, dm, dp_V, dm_V)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        self.final_matrix_ = self._overall(M)
        return self

    # ---------- Fit: K == 1 (binary) ----------
    def _fit_single_binary(self, DL, y):
        self._init_tracking()

        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)
        if L > 2:
            raise ValueError(f"Binary path requires 2 classes, got {L}")

        self._y = np.array(unique_labels, copy=True)
        self._w = np.zeros(L, dtype=int)

        overall = self._overall(M)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            self._w[l] = int(init_class_prototypes(
                overall, inClass, 1,
                base_seed=self.base_seed, label=lab, view_tag="global",
            )[0])

        closest_plus = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            closest_plus[inClass] = l
            closest_minus[inClass] = 1 - l

        rows = np.arange(self._m)
        dp = overall[rows, self._w[closest_plus]]
        dm = overall[rows, self._w[closest_minus]]
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]

        dp_V = M[:, rows, self._w[closest_plus]]
        dm_V = M[:, rows, self._w[closest_minus]]

        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        proto_losses = np.zeros(len(self._w))
        for _ in range(self.T):
            overall = self._overall(M)

            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                rf_plus = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta = 0.0
                best_i = None
                for i in rf_plus:
                    if i == self._w[k]:
                        continue

                    delta = 0.0
                    dp_new = overall[rf_plus, i]
                    mus_new = self.phi((dp_new - dm[rf_plus]) / (dp_new + dm[rf_plus] + 1e-5))
                    delta += np.sum(mus_new - mus[rf_plus])

                    dm_new = overall[rf_minus, i]
                    mus_new = self.phi((dp[rf_minus] - dm_new) / (dp[rf_minus] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[rf_minus])

                    if delta < best_delta:
                        best_delta = delta
                        best_i = i

                if best_i is None:
                    continue

                self._w[k] = best_i
                improved = True
                best_delta_global = best_delta

                dp[rf_plus] = overall[rf_plus, best_i]
                mus[rf_plus] = self.phi((dp[rf_plus] - dm[rf_plus]) / (dp[rf_plus] + dm[rf_plus] + 1e-5))
                dp_V[:, rf_plus] = M[:, rf_plus, best_i]

                dm[rf_minus] = overall[rf_minus, best_i]
                mus[rf_minus] = self.phi((dp[rf_minus] - dm[rf_minus]) / (dp[rf_minus] + dm[rf_minus] + 1e-5))
                dm_V[:, rf_minus] = M[:, rf_minus, best_i]

                expected_new = self._loss[-1] + best_delta
                actual_new = float(mus.sum())
                rel_err = abs(expected_new - actual_new) / (abs(self._loss[-1]) + 1e-12)
                if rel_err > 0.05:
                    print(f"[Warning] Loss deviation: expected {expected_new:.3f}, actual {actual_new:.3f} (rel_err={rel_err:.3%})")

                self._loss.append(actual_new)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                self._weights_update(dp, dm, dp_V, dm_V)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        self.final_matrix_ = self._overall(M)
        return self

    # ---------- Inference ----------
    def predict(self, DL):
        """
        DL: List of V test distance matrices (n x m) OR (n x total_K) to prototypes.
        Returns label array (n,).
        """
        Mtest = np.stack(DL, axis=0)
        D = np.tensordot(self._vWeights, Mtest, axes=(0, 0))
        if D.shape[1] == self._m:
            D = D[:, self._w]
        closest = np.argmin(D, axis=1)
        return self._y[closest]

    # ---------- Getters ----------
    def get_prototype_path(self):
        if self._w_history is None:
            return None
        return np.vstack(self._w_history) if len(self._w_history) else np.empty((0,))

    def get_vweight_path(self):
        """Path of the raw matrix factors a_v over training, shape (T, V).

        These are the internally tracked a_v with ||a||_2 = 1. For the
        interpretable group mixture a_v^2 use get_vweight_sq_path().
        Note: the sign of a_v is not identifiable (a_v and -a_v give the same
        distance); only a_v^2 is meaningful for interpretation.
        """
        if self._v_history is None:
            return None
        return np.vstack(self._v_history) if len(self._v_history) else np.empty((0,))

    def get_vweight_sq_path(self):
        """Path of the squared weights a_v^2 over training, shape (T, V).

        Each row sums to 1 and is the interpretable group mixture.
        """
        path = self.get_vweight_path()
        if path is None:
            return None
        return path ** 2 if path.size else path

    def get_vweights(self):
        """Final matrix weights as a dict with both parametrizations.

        Returns
        -------
        dict with keys:
          'a'      : final factors a_v        (||a||_2 = 1)
          'a_sq'   : final squared weights a_v^2 (sum = 1, interpretable mixture)
        """
        if not hasattr(self, "_vWeights") or self._vWeights is None:
            raise RuntimeError("Model has not been trained yet (fit has not been called).")
        a = np.array(self._vWeights, dtype=float, copy=True)
        return {"a": a, "a_sq": a ** 2}

    def get_training_log(self):
        if self._log is None:
            return None
        return {"loss": np.array(self._log["loss"], dtype=float)}