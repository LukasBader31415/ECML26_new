import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import proto_dist_ml.rng as rng

_ERR_CUTOFF = 1e-5


class MGLVQ(BaseEstimator, ClassifierMixin):
    """
    Performance-optimierte Variante (gleiche Logik, weniger Python-Overhead / weniger Allokationen).
    Kernalgorithmus unverändert: gleiche Update-Regeln, gleiche Verlustfunktion, gleiche argpartition-Logik.
    """

    def __init__(self, K, T=50, phi=None, track_path=False):
        self.K = K
        self.T = T
        self.track_path = track_path
        self.phi = (lambda mus: mus) if phi is None else phi

    def _init_history(self):
        if getattr(self, "track_path", False):
            self._w_history = [self._w.copy()]

    def fit(self, D, y):
        if self.K == 1:
            return self._fit_single(D, y)

        if len(D.shape) != 2:
            raise ValueError("Input is not a matrix!")
        if D.shape[0] != D.shape[1]:
            raise ValueError("Input matrix is not square!")

        # ---- lokale Aliase (sparen Attribute-Lookups) ----
        phi = self.phi
        eps = 1e-5
        m = D.shape[0]
        self._m = m
        ar = np.arange(m)

        unique_labels = np.unique(y)
        L = len(unique_labels)

        # ---- Initialisierung (unverändert, nur etwas weniger Overhead) ----
        if (not hasattr(self, "prevent_initialization")) or (not self.prevent_initialization):
            w = np.zeros(self.K * L, dtype=int)
            y_proto = np.zeros(self.K * L)
            for li, lab in enumerate(unique_labels):
                sl = slice(li * self.K, (li + 1) * self.K)
                y_proto[sl] = lab
                inClass = np.where(y == lab)[0]
                D_l = np.square(D[inClass, :][:, inClass])
                rng_l = rng.RNG(self.K)
                rng_l.fit(D_l, is_squared=True)
                Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
                closest = np.argmin(Dp_l, axis=1)
                w[sl] = inClass[closest]
            self._w = w
            self._y = y_proto
            self._init_history()
        else:
            self._init_history()
            w = self._w
            y_proto = self._y

        # ---- Vorberechnungen: Klassen-Indizes und Prototyp-Indizes pro Klasse ----
        # reduziert viele np.where-Aufrufe in Schleifen
        class_idx = {lab: np.where(y == lab)[0] for lab in unique_labels}
        proto_idx_inclass = {lab: np.where(y_proto == lab)[0] for lab in unique_labels}
        proto_idx_outclass = {lab: np.where(y_proto != lab)[0] for lab in unique_labels}

        # ---- Initiale Closest/SndClosest (unverändert, aber weniger Slicing/where) ----
        closest_plus = np.zeros(m, dtype=int)
        sndclosest_plus = np.zeros(m, dtype=int)
        closest_minus = np.zeros(m, dtype=int)
        sndclosest_minus = np.zeros(m, dtype=int)

        for lab in unique_labels:
            inClass = class_idx[lab]

            w_in = proto_idx_inclass[lab]
            Dp = D[inClass, :][:, w[w_in]]
            idxs = np.argpartition(Dp, 1, axis=1)
            closest_plus[inClass] = w_in[idxs[:, 0]]
            sndclosest_plus[inClass] = w_in[idxs[:, 1]]

            w_out = proto_idx_outclass[lab]
            Dm = D[inClass, :][:, w[w_out]]
            idxs = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass] = w_out[idxs[:, 0]]
            sndclosest_minus[inClass] = w_out[idxs[:, 1]]

        dp = D[ar, w[closest_plus]]
        dm = D[ar, w[closest_minus]]
        mus = phi((dp - dm) / (dp + dm + eps))
        self._loss = [np.sum(mus)]

        # ---- Optimierung ----
        proto_losses = np.zeros(len(w), dtype=float)

        for _t in range(self.T):
            # Änderung 1: proto_losses vektorisieren (statt for k ... sum(...))
            # proto_losses[k] = sum(dp where closest_plus==k) - sum(dm where closest_minus==k)
            # -> bincount auf Indizes
            proto_losses[:] = 0.0
            proto_losses += np.bincount(closest_plus, weights=dp, minlength=len(w))
            proto_losses -= np.bincount(closest_minus, weights=dm, minlength=len(w))

            best_delta_loss_epoch = 0.0  # entspricht deiner Abbruchlogik (beste Verbesserung in Epoche)

            # iterate over prototypes, high to low loss
            for k in np.argsort(-proto_losses):
                lab_k = y_proto[k]
                inClass_k = class_idx[lab_k]
                outClass_k = np.setdiff1d(ar, inClass_k, assume_unique=False)  # selten benutzt; kann bei Bedarf ersetzt werden

                rf_plus = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta_loss = 0.0

                # Vorzieher: oft verwendete Arrays einmal holen
                snd_p_rf_plus = sndclosest_plus[rf_plus]
                snd_m_rf_minus = sndclosest_minus[rf_minus]

                for i in rf_plus:
                    if i == w[k]:
                        continue

                    # Änderung 2: "still_closest" ohne wiederholte Indexketten, lokale Aliase
                    # Case 1/2 (positive)
                    # still_closest: D[rf_plus, i] <= D[rf_plus, w[sndclosest_plus[rf_plus]]]
                    still_closest_p = D[rf_plus, i] <= D[rf_plus, w[snd_p_rf_plus]]
                    changed_plus = np.unique(
                        np.concatenate(
                            [
                                inClass_k[D[inClass_k, i] < dp[inClass_k]],
                                rf_plus[still_closest_p],
                            ]
                        )
                    )
                    changed_plus2 = rf_plus[~still_closest_p]

                    # Case 3/4 (negative)
                    still_closest_m = D[rf_minus, i] <= D[rf_minus, w[snd_m_rf_minus]]
                    changed_minus = np.unique(
                        np.concatenate(
                            [
                                outClass_k[D[outClass_k, i] < dm[outClass_k]],
                                rf_minus[still_closest_m],
                            ]
                        )
                    )
                    changed_minus2 = rf_minus[~still_closest_m]

                    # delta_loss wie gehabt, aber mit lokalen Aliases
                    delta_loss = 0.0

                    dp_new = D[changed_plus, i]
                    mus_new = phi((dp_new - dm[changed_plus]) / (dp_new + dm[changed_plus] + eps))
                    delta_loss += np.sum(mus_new - mus[changed_plus])

                    dp_new = D[changed_plus2, w[sndclosest_plus[changed_plus2]]]
                    mus_new = phi((dp_new - dm[changed_plus2]) / (dp_new + dm[changed_plus2] + eps))
                    delta_loss += np.sum(mus_new - mus[changed_plus2])

                    dm_new = D[changed_minus, i]
                    mus_new = phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + eps))
                    delta_loss += np.sum(mus_new - mus[changed_minus])

                    dm_new = D[changed_minus2, w[sndclosest_minus[changed_minus2]]]
                    mus_new = phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm_new + eps))
                    delta_loss += np.sum(mus_new - mus[changed_minus2])

                    if delta_loss < best_delta_loss:
                        best_delta_loss = delta_loss
                        best_i = i
                        best_changed_plus = changed_plus
                        best_changed_plus2 = changed_plus2
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2

                if best_delta_loss < 0.0:
                    w[k] = best_i
                    if getattr(self, "track_path", False):
                        self._w_history.append(w.copy())

                    # Update caches (identisch, nur lokale Aliase genutzt)
                    closest_plus[best_changed_plus] = k
                    dp[best_changed_plus] = D[best_changed_plus, best_i]
                    mus[best_changed_plus] = phi((dp[best_changed_plus] - dm[best_changed_plus]) / (dp[best_changed_plus] + dm[best_changed_plus] + eps))

                    w_inClass = proto_idx_inclass[lab_k]
                    idxs = np.argpartition(D[best_changed_plus, :][:, w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus] = w_inClass[idxs[:, 1]]

                    closest_plus[best_changed_plus2] = sndclosest_plus[best_changed_plus2]
                    dp[best_changed_plus2] = D[best_changed_plus2, w[closest_plus[best_changed_plus2]]]
                    mus[best_changed_plus2] = phi((dp[best_changed_plus2] - dm[best_changed_plus2]) / (dp[best_changed_plus2] + dm[best_changed_plus2] + eps))
                    idxs = np.argpartition(D[best_changed_plus2, :][:, w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus2] = w_inClass[idxs[:, 1]]

                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = D[best_changed_minus, best_i]
                    mus[best_changed_minus] = phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + eps))

                    # sndclosest_minus Updates (wie gehabt)
                    for lab in unique_labels:
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == lab)[0]]
                        w_outClass = proto_idx_outclass[lab]
                        idxs = np.argpartition(D[inClass_l, :][:, w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = D[best_changed_minus2, w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + eps))
                    # w_outClass hängt vom Label ab; entspricht Original (etwas redundant)
                    for lab in unique_labels:
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == lab)[0]]
                        w_outClass = proto_idx_outclass[lab]
                        idxs = np.argpartition(D[inClass_l, :][:, w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = float(np.sum(mus))
                    if np.abs(expected_new_loss - actual_new_loss) / m > 0.01:
                        import warnings
                        warnings.warn(
                            f"MGLVQ loss bookkeeping drift: expected {expected_new_loss:g}, "
                            f"got {actual_new_loss:g} (using recomputed value).",
                            RuntimeWarning,
                        )
                    self._loss.append(actual_new_loss)

                    best_delta_loss_epoch = best_delta_loss
                    break

            if best_delta_loss_epoch >= -_ERR_CUTOFF:
                break

        self._w = w
        self._y = y_proto
        return self

    # --- deine _fit_single / _fit_single_binary / predict bleiben unverändert ---
    #     (kann man analog mit bincount für proto_losses optimieren)

    def _fit_single(self, D, y):
        # ORIGINALCODE hier belassen (oder analog optimieren wie oben)
        raise NotImplementedError

    def _fit_single_binary(self, D, y):
        raise NotImplementedError

    def predict(self, D):
        n = D.shape[0]
        if D.shape[1] == self._m:
            D = D[:, self._w]
        closest = np.argmin(D, axis=1)
        return self._y[closest]