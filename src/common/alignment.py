from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class AlignedData:
    used_ids: pd.Index
    y: np.ndarray
    matrices: Dict[str, pd.DataFrame]


def _validate_matrix(D: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(D, pd.DataFrame):
        raise TypeError(f"{name} ist kein pandas DataFrame.")

    D = D.copy()
    D.index = D.index.astype(str)
    D.columns = D.columns.astype(str)

    if D.shape[0] != D.shape[1]:
        raise ValueError(f"{name} ist nicht quadratisch: {D.shape}")

    if not D.index.equals(D.columns):
        missing_in_cols = set(D.index) - set(D.columns)
        missing_in_idx = set(D.columns) - set(D.index)
        raise ValueError(
            f"{name}: Index und Columns stimmen nicht überein. "
            f"Nur Index: {len(missing_in_cols)}, nur Columns: {len(missing_in_idx)}"
        )

    if D.index.has_duplicates or D.columns.has_duplicates:
        raise ValueError(f"{name}: doppelte IDs in Index oder Columns.")

    return D


def align_views_and_target(
    matrices: Dict[str, pd.DataFrame],
    y: pd.Series,
    order: str = "y",
) -> AlignedData:
    """
    Schneidet alle Matrizen und y auf gemeinsame IDs und richtet alles gleich aus.

    Parameters
    ----------
    matrices : dict[str, pd.DataFrame]
        Dissimilarity-Matrizen mit ID-Index und ID-Spalten.
    y : pd.Series
        Zielvariable mit CustomerCode als Index.
    order : str
        "y" = Reihenfolge aus y beibehalten.
        "sorted" = IDs lexikographisch sortieren.

    Returns
    -------
    AlignedData
    """
    if not isinstance(y, pd.Series):
        raise TypeError("y muss eine pandas Series sein.")

    y = y.copy()
    y.index = y.index.astype(str)

    if y.index.has_duplicates:
        dup = y.index[y.index.duplicated()].tolist()[:10]
        raise ValueError(f"y hat doppelte IDs, z.B.: {dup}")

    validated = {}
    common_ids = set(y.index)

    for name, D in matrices.items():
        Dv = _validate_matrix(D, name=name)
        validated[name] = Dv
        common_ids &= set(Dv.index)

    if not common_ids:
        raise ValueError("Keine gemeinsamen IDs zwischen y und Matrizen gefunden.")

    if order == "y":
        used_ids = pd.Index([idx for idx in y.index if idx in common_ids], dtype="object")
    elif order == "sorted":
        used_ids = pd.Index(sorted(common_ids), dtype="object")
    else:
        raise ValueError("order muss 'y' oder 'sorted' sein.")

    y_aligned = y.loc[used_ids].to_numpy(dtype=int)

    matrices_aligned = {
        name: D.loc[used_ids, used_ids]
        for name, D in validated.items()
    }

    return AlignedData(
        used_ids=used_ids,
        y=y_aligned,
        matrices=matrices_aligned,
    )
