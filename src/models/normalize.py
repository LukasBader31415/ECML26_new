import numpy as np
import pandas as pd


def normalize_views(
    D_dict: dict[str, np.ndarray | pd.DataFrame],
    q_low: float = 5,
    q_high: float = 95,
) -> dict[str, np.ndarray]:
    """
    Normiert mehrere Distanzmatrizen robust auf [0,1].
    Quantile werden aus den Off-Diagonalwerten berechnet.
    """
    normalized = {}

    for name, D in D_dict.items():
        D = np.asarray(D, dtype=float).copy()

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"{name}: Matrix muss quadratisch sein, got {D.shape}")

        mask = np.triu(np.ones_like(D, dtype=bool), k=1)
        vals = D[mask]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            raise ValueError(f"{name}: keine gültigen Off-Diagonalwerte gefunden.")

        lo = np.percentile(vals, q_low)
        hi = np.percentile(vals, q_high)

        if np.isclose(hi, lo):
            Dn = np.zeros_like(D)
        else:
            Dn = (D - lo) / (hi - lo)
            Dn = np.clip(Dn, 0.0, 1.0)

        np.fill_diagonal(Dn, 0.0)
        normalized[name] = Dn

    return normalized


def build_m3glvq_view_dict(
    matrices: dict[str, pd.DataFrame],
    view_name_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Wandelt aligned matrices in das M3GLVQ-View-Dict um.
    Default-Mapping:
        naics -> NAICS
        hs    -> HS
        am    -> AM
    """
    if view_name_map is None:
        view_name_map = {
            "naics": "NAICS",
            "hs": "HS",
            "am": "AM",
        }

    out = {}
    for src_name, target_name in view_name_map.items():
        if src_name not in matrices:
            raise ValueError(f"Fehlende View '{src_name}' in matrices.")
        out[target_name] = np.asarray(matrices[src_name], dtype=float)

    return out


def build_normalized_view_list(
    matrices: dict[str, pd.DataFrame],
    view_order: tuple[str, ...] = ("NAICS", "HS", "AM"),
    q_low: float = 5,
    q_high: float = 95,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """
    Baut direkt die geordnete Liste DL für M3GLVQ.

    Rückgabe:
        DL, normalized_dict
    """
    D_views = build_m3glvq_view_dict(matrices)
    D_views_norm = normalize_views(D_views, q_low=q_low, q_high=q_high)

    missing = [name for name in view_order if name not in D_views_norm]
    if missing:
        raise ValueError(f"Fehlende normalisierte Views: {missing}")

    DL = [D_views_norm[name] for name in view_order]
    return DL, D_views_norm
