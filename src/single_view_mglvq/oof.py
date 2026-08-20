import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def run_oof_predictions(
    D_df: pd.DataFrame,
    y_arr: np.ndarray,
    ids: np.ndarray,
    model_cls: type,
    params: dict,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    D = D_df.to_numpy()
    y = np.asarray(y_arr, dtype=int).ravel()
    ids = np.asarray(ids, dtype=str)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(y)), y), start=1):
        D_tr = D[np.ix_(tr_idx, tr_idx)]
        D_va_tr = D[np.ix_(va_idx, tr_idx)]

        clf = model_cls(**params)
        clf.fit(D_tr, y[tr_idx])
        pred = clf.predict(D_va_tr).astype(int)

        for j, idx in enumerate(va_idx):
            rows.append({
                "id": ids[idx],
                "fold": fold,
                "y_true": int(y[idx]),
                "y_pred": int(pred[j]),
                "correct": int(pred[j] == y[idx]),
            })

    return pd.DataFrame(rows)
