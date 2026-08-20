"""
Data loading for the ECML26 pipeline — product 0601, the validated 2025 aligned
customers (classes {0: 987, 1: 1038}).

Faithful to the old pipeline: three dissimilarity views (NAICS / HS / AM),
alignment ``order="y"`` (keep the label file's customer order among common IDs),
robust q5/q95 off-diagonal normalization via ``build_normalized_view_list``.
No pre-screening — that is a separate extension and stays out here.

Two modes, one entry point ``load_aligned``:

  * repo layout (real run on JupyterHub):
        load_aligned(base_path="…/data/raw/…", product="0601")
    expects ``dissimilarity_*.pkl`` matrices and ``{product}.pkl`` target in
    ``base_path`` (matching src/data_loading conventions).

  * explicit flat files (local testing, e.g. the audited uploads):
        load_aligned(
            matrix_paths={"naics": ".../dissimilarity_naics_matrix_3_.pkl",
                          "hs":    ".../dissimilarity_hs_matrix_2_.pkl",
                          "am":    ".../dissimilarity_am_A_matrix_2_.pkl"},
            target_path=".../A_2_.pkl", target_col="A")
    (the uploaded target file carries its label in column ``"A"``, not ``"0601"``.)

Returns ``(codes, y, DL)`` with ``codes`` the aligned CustomerCodes (str array),
``y`` int labels, ``DL`` the normalized view list in ``view_order``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .common.alignment import align_views_and_target
from .models.normalize import build_normalized_view_list

VIEW_NAMES = ("NAICS", "HS", "AM")
VIEW_LABELS = {
    "NAICS": "Industry (NAICS)",
    "HS": "Products (HS)",
    "AM": "Applications (AM)",
}
# canonical internal keys for the three matrices (feed build_m3glvq_view_dict)
_MATRIX_KEYS = ("naics", "hs", "am")


def _matrix_key_from_name(name: str) -> str | None:
    """Map a file stem / path to one of naics|hs|am by substring."""
    low = str(name).lower()
    for key in _MATRIX_KEYS:
        if key in low:
            return key
    return None


def _load_matrices_repo(base_path: Path) -> dict[str, pd.DataFrame]:
    """Load dissimilarity_*.pkl from a directory, keyed naics|hs|am."""
    out: dict[str, pd.DataFrame] = {}
    for file in sorted(base_path.glob("dissimilarity_*.pkl")):
        key = _matrix_key_from_name(file.stem)
        if key is None:
            continue
        if key in out:
            raise ValueError(
                f"Mehrere Matrizen fuer View '{key}' gefunden (z.B. {file.name}). "
                "Ordner enthaelt Duplikate — matrix_paths explizit setzen."
            )
        out[key] = pd.read_pickle(file)
    missing = [k for k in _MATRIX_KEYS if k not in out]
    if missing:
        raise ValueError(f"Fehlende Views in {base_path}: {missing}")
    return out


def _load_matrices_explicit(matrix_paths: dict[str, str | Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key in _MATRIX_KEYS:
        if key not in matrix_paths:
            raise ValueError(f"matrix_paths fehlt View '{key}'.")
        out[key] = pd.read_pickle(matrix_paths[key])
    return out


def _load_target_series(
    base_path: Path | None,
    product: str,
    target_path: str | Path | None,
    target_col: str | None,
) -> pd.Series:
    """Return the label as a Series indexed by CustomerCode (str)."""
    if target_path is not None:
        df = pd.read_pickle(target_path).copy()
        col = target_col or product
    else:
        if base_path is None:
            raise ValueError("Entweder base_path oder target_path angeben.")
        df = pd.read_pickle(Path(base_path) / f"{product}.pkl").copy()
        col = target_col or product

    if "CustomerCode" not in df.columns:
        raise ValueError("Ziel-DataFrame ohne Spalte 'CustomerCode'.")
    if col not in df.columns:
        raise ValueError(
            f"Label-Spalte '{col}' nicht im Ziel-DataFrame (Spalten: {list(df.columns)}). "
            "Strikter Match, kein Fallback."
        )
    df["CustomerCode"] = df["CustomerCode"].astype(str)
    s = pd.Series(df[col].astype(int).values, index=df["CustomerCode"].values, name="y")
    if s.index.has_duplicates:
        raise ValueError("Doppelte CustomerCodes im Ziel-DataFrame.")
    return s


def load_aligned(
    base_path: str | Path | None = None,
    product: str = "0601",
    *,
    matrix_paths: dict[str, str | Path] | None = None,
    target_path: str | Path | None = None,
    target_col: str | None = None,
    order: str = "y",
    q_low: float = 5,
    q_high: float = 95,
    view_order: tuple[str, ...] = VIEW_NAMES,
    subsample: int | None = None,
    seed: int = 0,
):
    """Load, align (order='y') and q5/q95-normalize the three views for one product.

    Returns
    -------
    codes : np.ndarray[str]   aligned CustomerCodes
    y     : np.ndarray[int]   labels
    DL    : list[np.ndarray]  normalized views in ``view_order`` (NAICS, HS, AM)
    """
    base_path = Path(base_path) if base_path is not None else None

    matrices = (
        _load_matrices_explicit(matrix_paths)
        if matrix_paths is not None
        else _load_matrices_repo(base_path)
    )
    y_series = _load_target_series(base_path, product, target_path, target_col)

    aligned = align_views_and_target(matrices, y_series, order=order)
    # aligned.matrices keeps the naics|hs|am keys; build_normalized_view_list
    # maps them to NAICS|HS|AM and applies q5/q95.
    DL, _ = build_normalized_view_list(
        aligned.matrices, view_order=view_order, q_low=q_low, q_high=q_high
    )
    codes = np.asarray(aligned.used_ids, dtype=str)
    y = np.asarray(aligned.y, dtype=int)

    if subsample is not None and subsample < len(y):
        rs = np.random.RandomState(seed)
        idx = np.sort(rs.choice(len(y), subsample, replace=False))
        DL = [D[np.ix_(idx, idx)] for D in DL]
        y = y[idx]
        codes = codes[idx]

    return codes, y, DL


@dataclass
class Bundle:
    """Everything the pipeline blocks need, all on one aligned customer order.

    codes         : np.ndarray[str]              aligned CustomerCodes
    y             : np.ndarray[int]              labels
    y_series      : pd.Series (index=codes)      labels for the structural/single-view runners
    matrices_raw  : dict[str, pd.DataFrame]      RAW aligned matrices (keys 'naics'|'hs'|'am')
                                                 -> structural block + single-view MGLVQ
    matrices_norm : dict[str, np.ndarray]        q5/q95-normalized (keys 'NAICS'|'HS'|'AM')
                                                 -> distance-distribution table
    DL            : list[np.ndarray]             normalized views in view_order -> engine / search / rOOF
    """
    codes: np.ndarray
    y: np.ndarray
    y_series: pd.Series
    matrices_raw: dict
    matrices_norm: dict
    DL: list


def load_bundle(
    base_path: str | Path | None = None,
    product: str = "0601",
    *,
    matrix_paths: dict[str, str | Path] | None = None,
    target_path: str | Path | None = None,
    target_col: str | None = None,
    order: str = "y",
    q_low: float = 5,
    q_high: float = 95,
    view_order: tuple[str, ...] = VIEW_NAMES,
    subsample: int | None = None,
    seed: int = 0,
) -> Bundle:
    """Align once, derive everything. Use this from the notebook so all blocks
    share an identical customer order (order='y')."""
    base_path = Path(base_path) if base_path is not None else None
    matrices = (
        _load_matrices_explicit(matrix_paths)
        if matrix_paths is not None
        else _load_matrices_repo(base_path)
    )
    y_series_in = _load_target_series(base_path, product, target_path, target_col)

    aligned = align_views_and_target(matrices, y_series_in, order=order)
    matrices_raw = {k: v for k, v in aligned.matrices.items()}   # naics|hs|am DataFrames
    DL, matrices_norm = build_normalized_view_list(
        matrices_raw, view_order=view_order, q_low=q_low, q_high=q_high
    )
    codes = np.asarray(aligned.used_ids, dtype=str)
    y = np.asarray(aligned.y, dtype=int)

    if subsample is not None and subsample < len(y):
        rs = np.random.RandomState(seed)
        idx = np.sort(rs.choice(len(y), subsample, replace=False))
        keep = codes[idx]
        matrices_raw = {k: D.loc[keep, keep] for k, D in matrices_raw.items()}
        matrices_norm = {k: D[np.ix_(idx, idx)] for k, D in matrices_norm.items()}
        DL = [D[np.ix_(idx, idx)] for D in DL]
        y = y[idx]; codes = keep

    y_series = pd.Series(y, index=pd.Index(codes, name="CustomerCode"), name="y")
    return Bundle(codes=codes, y=y, y_series=y_series,
                  matrices_raw=matrices_raw, matrices_norm=matrices_norm, DL=DL)


def synthetic_bundle(m: int = 120, seed: int = 0, view_order: tuple[str, ...] = VIEW_NAMES) -> Bundle:
    """Small structured synthetic bundle for USE_SYNTHETIC end-to-end smoke tests.

    Two classes separated in a latent space; the three views differ in how much
    class signal they carry (naics strongest, am weakest) so the pipeline
    reproduces the qualitative NAICS-dominance without any real data.
    """
    rng = np.random.RandomState(seed)
    y = np.array([0] * (m // 2) + [1] * (m - m // 2))
    rng.shuffle(y)
    codes = np.array([f"C{idx:05d}" for idx in range(m)], dtype=str)

    # per-view latent separation (larger gap -> more class structure)
    gaps = {"naics": 2.4, "hs": 1.3, "am": 0.6}
    dims = 8
    matrices_raw, matrices_norm = {}, {}
    view_key = {"NAICS": "naics", "HS": "hs", "AM": "am"}

    for vname in view_order:
        key = view_key[vname]
        centers = rng.randn(2, dims)
        X = rng.randn(m, dims) + gaps[key] * centers[y]
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(D, 0.0)
        df = pd.DataFrame(D, index=codes, columns=codes)
        matrices_raw[key] = df

    DL, matrices_norm = build_normalized_view_list(
        matrices_raw, view_order=view_order, q_low=5, q_high=95
    )
    y_series = pd.Series(y, index=pd.Index(codes, name="CustomerCode"), name="y")
    return Bundle(codes=codes, y=y, y_series=y_series,
                  matrices_raw=matrices_raw, matrices_norm=matrices_norm, DL=DL)
