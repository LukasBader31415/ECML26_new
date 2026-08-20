import time
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.notebook import tqdm

from src.single_view_mglvq.oof import run_oof_predictions
from src.single_view_mglvq.metrics import balanced_accuracy_from_oof


def grid_search_best(
    D_df: pd.DataFrame,
    y_arr,
    ids,
    model_cls: type,
    param_grid: dict,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> tuple[dict, float, pd.DataFrame]:
    if not param_grid:
        raise ValueError("param_grid fehlt.")
    plist = list(ParameterGrid(param_grid))
    if not plist:
        raise ValueError("param_grid hat 0 Kombinationen.")

    rows = []
    best_score = -1e18
    best_params = None

    iterator = plist
    if show_progress:
        iterator = tqdm(
            plist,
            total=len(plist),
            desc=progress_desc or "Grid Search",
            leave=True,
        )

    for params in iterator:
        t0 = time.time()

        oof = run_oof_predictions(
            D_df=D_df,
            y_arr=y_arr,
            ids=ids,
            model_cls=model_cls,
            params=params,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        score = balanced_accuracy_from_oof(oof)
        runtime_sec = time.time() - t0

        rows.append({
            "score": score,
            "runtime_sec": runtime_sec,
            **params,
        })

        if score > best_score:
            best_score = score
            best_params = dict(params)

        if show_progress:
            iterator.set_postfix({
                "score": f"{score:.4f}",
                "best": f"{best_score:.4f}",
                "K": params.get("K", "-"),
                "sec": f"{runtime_sec:.1f}",
            })

    if best_params is None:
        raise ValueError("Keine besten Parameter gefunden.")

    grid_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, float(best_score), grid_df