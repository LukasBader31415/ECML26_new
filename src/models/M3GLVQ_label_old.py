__author__ = 'Lukas Bader, Dietlind Zühlke'
__copyright__ = ('Copyright 2019, Benjamin Paaßen; '
                 'Copyright 2026, Lukas Bader, Dietlind Zühlke')
__license__ = 'GPLv3'
__version__ = '2.1.0-label'
__maintainer__ = 'Lukas Bader'
__email__ = 'lukas.bader@pferd.com'


"""
M3GLVQ -- LABEL-SPECIFIC variant (one squared-weight vector per label)

This module implements the M3GLVQ algorithm as described in:

    Lukas Bader, Ina Terwey-Scheulen, Dietlind Zühlke,
    "Implementation of Multi-Matrix Median Generalized Learning Vector Quantization",
    ESANN 2026.

The implementation is based on and extends the original MGLVQ code by
Benjamin Paaßen (proto-dist-ml, GNU GPLv3).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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


class M3GLVQ_Label(BaseEstimator, ClassifierMixin):
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
        v_init=None
    ):
        self.K = K
        self.T = T
        self.phi = (lambda mus: mus) if phi is None else phi

        self.track_path = bool(track_path)
        self.track_vweights = bool(track_vweights)
        self.track_metrics = bool(track_metrics)
        self.eta = float(eta)
        self.v_init = None if v_init is None else np.asarray(v_init, dtype=float)

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

    # ===================================================================
    # ===============  LABEL-SPECIFIC WEIGHTING  ========================
    # ===================================================================
    # Each label l owns its own factor vector a^(l) (length V, ||a^(l)||_2 = 1).
    # The combined distance for label l is  D*^(l) = Σ_v (a^(l)_v)^2 D^(v).
    # Simplified scheme: every data point i is evaluated entirely through the
    # metric of its OWN label, i.e. both d_i^+ and d_i^- use a^(y_i). Hence the
    # gradient w.r.t. a^(l) only collects points with y_i == l (no cross-coupling).

    def _overall_for_label(self, M, l):
        """Squared-weighted distance matrix for label index l: Σ_v a^(l)_v^2 D^(v)."""
        return np.tensordot(self._vWeights_ls[l] ** 2, M, axes=(0, 0))

    def _weights_update_label(self, l, dp, dm, dp_V, dm_V, mask):
        """
        Gradient step for the factor vector a^(l) of a single label l.

        Only points with y_i == l contribute (mask selects them). The per-point
        contribution is the same squared-parametrization gradient as in the
        global path:

            dCF/da^(l)_v = Σ_{i: y_i=l} 4 a^(l)_v (d_iv^+ d_i^- - d_iv^- d_i^+)
                                                  / (d_i^+ + d_i^-)^2

        dp, dm     : (m,)    combined d^+ , d^-      (only mask entries used)
        dp_V, dm_V : (V, m)  per-matrix d_iv^+, d_iv^-
        mask       : (m,) bool, True where y_i == l
        """
        if not np.any(mask):
            return
        a_l = self._vWeights_ls[l]
        den = (dp[mask] + dm[mask] + 1e-5) ** 2                  # (n_l,)
        num = dp_V[:, mask] * dm[mask][None, :] \
              - dm_V[:, mask] * dp[mask][None, :]                # (V, n_l)
        grad = (4.0 * a_l[:, None] * num / den[None, :]).sum(axis=1)   # (V,)

        g_norm = np.linalg.norm(grad, 2) + 1e-12
        step = grad / g_norm
        a_new = a_l - self.eta * step
        self._vWeights_ls[l] = _normalize_l2(a_new)              # Σ a^(l)_v^2 = 1

    def fit(self, DL, y):
        """
        Train M3GLVQ with ONE squared-weight vector a^(l) per label.

        Mirrors the general (arbitrary-K) median-LVQ prototype optimization, but
        keeps an (L, V) weight matrix instead of a single (V,) vector. After each
        accepted prototype swap, every label's weight vector is updated by one
        gradient step and L2-normalized.

        DL : list of V distance matrices, each (m, m)
        y  : (m,) label array
        """
        self._init_tracking()
        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)

        # ---- per-label weight matrix (L, V), each row L2-normalized ----
        # v_init_per_label: dict {label: a-vector} -> eigene Brille je Label
        #   (zum Einfrieren label-spezifischer Gewichte mit eta=0)
        # v_init: ein Vektor -> auf alle Label kopiert
        if getattr(self, 'v_init_per_label', None) is not None:
            self._vWeights_ls = np.zeros((L, self._V))
            for li, lab in enumerate(unique_labels):
                self._vWeights_ls[li] = _normalize_l2(
                    np.asarray(self.v_init_per_label[lab], dtype=float))
        elif self.v_init is not None:
            base = _normalize_l2(self.v_init)
            self._vWeights_ls = np.tile(base, (L, 1))
        else:
            self._vWeights_ls = np.full((L, self._V), 1.0 / np.sqrt(self._V))
        self._ls_labels = np.array(unique_labels, copy=True)

        K_per = self._resolve_K_per_label(unique_labels)
        total_K = int(K_per.sum())
        self._y = np.repeat(unique_labels, K_per)
        self._w = np.zeros(total_K, dtype=int)

        # label index of each data point and of each prototype
        lab_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_idx = np.array([lab_to_idx[v] for v in y])
        proto_lab_idx = np.array([lab_to_idx[v] for v in self._y])

        # ---- prototype initialization (robust k-medoid-style, per label) ----
        # The original RNG initializer from proto_dist_ml is brittle for small
        # per-class sizes; we use a simple, deterministic medoid scheme instead:
        # pick the k points with the smallest summed squared distance, then
        # round-robin-assign the remaining own-class points to their nearest pick
        # so that all k medoids are seeded apart.
        if (not hasattr(self, 'prevent_initialization')) or (not self.prevent_initialization):
            offset = 0
            for l, lab in enumerate(unique_labels):
                k_l = int(K_per[l])
                idx_w = np.arange(offset, offset + k_l)
                inClass = np.where(y == lab)[0]
                overall_l = self._overall_for_label(M, l)
                D_l = np.square(overall_l[inClass, :][:, inClass])  # (n_l, n_l)
                # first medoid: minimal total squared distance
                chosen = [int(np.argmin(D_l.sum(axis=1)))]
                # remaining medoids: farthest-point seeding
                for _ in range(1, k_l):
                    dmin = D_l[:, chosen].min(axis=1)
                    chosen.append(int(np.argmax(dmin)))
                self._w[idx_w] = inClass[np.array(chosen[:k_l])]
                offset += k_l

        rows = np.arange(self._m)

        def _assign():
            """Recompute closest +/- prototypes and d^+, d^- using per-label metrics.

            Each point i is scored entirely under the metric of its own label.
            """
            closest_plus = np.zeros(self._m, dtype=int)
            closest_minus = np.zeros(self._m, dtype=int)
            for l, lab in enumerate(unique_labels):
                inClass = np.where(y == lab)[0]
                overall_l = self._overall_for_label(M, l)   # metric of label l
                in_w = np.where(self._y == lab)[0]
                out_w = np.where(self._y != lab)[0]
                Dp = overall_l[inClass, :][:, self._w[in_w]]
                closest_plus[inClass] = in_w[np.argmin(Dp, axis=1)]
                Dm = overall_l[inClass, :][:, self._w[out_w]]
                closest_minus[inClass] = out_w[np.argmin(Dm, axis=1)]
            # d^+, d^- and per-matrix parts, each point under its own-label metric
            dp = np.zeros(self._m); dm = np.zeros(self._m)
            dp_V = np.zeros((self._V, self._m))
            dm_V = np.zeros((self._V, self._m))
            for l, lab in enumerate(unique_labels):
                inClass = np.where(y == lab)[0]
                overall_l = self._overall_for_label(M, l)
                dp[inClass] = overall_l[inClass, self._w[closest_plus[inClass]]]
                dm[inClass] = overall_l[inClass, self._w[closest_minus[inClass]]]
                dp_V[:, inClass] = M[:, inClass, self._w[closest_plus[inClass]]]
                dm_V[:, inClass] = M[:, inClass, self._w[closest_minus[inClass]]]
            return closest_plus, closest_minus, dp, dm, dp_V, dm_V

        closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign()
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]
        self._snapshot(w=self._w, v=self._vWeights_ls.copy(), loss=self._loss[-1])

        # ---- optimization loop: greedy prototype swap + per-label weight step ----
        for _ in range(self.T):
            proto_losses = np.zeros(total_K)
            for k in range(total_K):
                proto_losses[k] = (np.sum(dp[closest_plus == k])
                                   - np.sum(dm[closest_minus == k]))

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                lab_k = self._y[k]
                l_k = lab_to_idx[lab_k]
                overall_k = self._overall_for_label(M, l_k)
                rf_plus = np.where(closest_plus == k)[0]
                inClass_k = np.where(y == lab_k)[0]

                best_delta = 0.0
                best_i = None
                for i in rf_plus:
                    if i == self._w[k]:
                        continue
                    # candidate: move prototype k to data index i
                    # affected own-class points: those whose d^+ would drop
                    cand_dp = overall_k[inClass_k, i]
                    cp = inClass_k[cand_dp < dp[inClass_k]]
                    new_dp = overall_k[cp, i]
                    mus_new = self.phi((new_dp - dm[cp]) / (new_dp + dm[cp] + 1e-5))
                    delta = float(np.sum(mus_new - mus[cp]))
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i

                if best_i is None:
                    continue

                self._w[k] = best_i
                improved = True
                best_delta_global = best_delta
                # full reassignment keeps the bookkeeping simple and correct
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign()
                mus = self.phi((dp - dm) / (dp + dm + 1e-5))
                self._loss.append(float(mus.sum()))
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(),
                               loss=self._loss[-1])

                # one gradient step for every label's weight vector
                for l, lab in enumerate(unique_labels):
                    mask = (y == lab)
                    self._weights_update_label(l, dp, dm, dp_V, dm_V, mask)
                # reassignment + loss after the weight change
                closest_plus, closest_minus, dp, dm, dp_V, dm_V = _assign()
                mus = self.phi((dp - dm) / (dp + dm + 1e-5))
                self._loss.append(float(mus.sum()))
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(),
                               loss=self._loss[-1])
                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights_ls.copy(),
                               loss=self._loss[-1])
                break

        # final per-label combined matrices
        self.final_matrix_ls_ = {
            unique_labels[l]: self._overall_for_label(M, l) for l in range(L)
        }
        self.final_matrix_ = self.final_matrix_ls_[unique_labels[0]]
        return self

    def get_vweights(self):
        """Final per-label weights as a dict {label: {'a':..., 'a_sq':...}}."""
        if not hasattr(self, "_vWeights_ls") or self._vWeights_ls is None:
            raise RuntimeError("fit_label_specific has not been called yet.")
        out = {}
        for l, lab in enumerate(self._ls_labels):
            a = np.array(self._vWeights_ls[l], dtype=float, copy=True)
            out[lab] = {"a": a, "a_sq": a ** 2}
        return out

    def predict(self, DL):
        """
        Predict labels using the per-label metrics.

        For each label l the combined distance to that label's prototypes is
        computed with a^(l); a point is assigned to the label whose nearest
        prototype (under that label's own metric) is closest.
        """
        Mtest = np.stack(DL, axis=0)
        n = Mtest.shape[1]
        best_d = np.full(n, np.inf)
        best_lab = np.empty(n, dtype=self._ls_labels.dtype)
        for l, lab in enumerate(self._ls_labels):
            D = np.tensordot(self._vWeights_ls[l] ** 2, Mtest, axes=(0, 0))
            if D.shape[1] == self._m:
                D = D[:, self._w]
            cols = np.where(self._y == lab)[0]
            d_lab = D[:, cols].min(axis=1)
            upd = d_lab < best_d
            best_d[upd] = d_lab[upd]
            best_lab[upd] = lab
        return best_lab

    # ---------- Getters ----------
    def get_prototype_path(self):
        if self._w_history is None:
            return None
        return np.vstack(self._w_history) if len(self._w_history) else np.empty((0,))

    def get_vweight_path(self):
        """Path of the raw per-label factors a^(l)_v over training.

        Returns an array of shape (T, L, V): for each tracked step, the full
        (L, V) weight matrix with each row L2-normalized (||a^(l)||_2 = 1).
        For the interpretable group mixture a^(l)_v^2 use get_vweight_sq_path().
        Note: the sign of a^(l)_v is not identifiable; only a^(l)_v^2 is
        meaningful for interpretation.
        """
        if self._v_history is None:
            return None
        return np.stack(self._v_history, axis=0) if len(self._v_history) \
            else np.empty((0,))

    def get_vweight_sq_path(self):
        """Path of the squared per-label weights a^(l)_v^2, shape (T, L, V).

        Each row (over V) sums to 1 and is the interpretable group mixture
        for that label.
        """
        path = self.get_vweight_path()
        if path is None:
            return None
        return path ** 2 if path.size else path

    def get_training_log(self):
        if self._log is None:
            return None
        return {"loss": np.array(self._log["loss"], dtype=float)}
