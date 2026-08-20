from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from src.common.alignment import align_views_and_target
from src.single_view_mglvq.gridsearch import grid_search_best
from src.single_view_mglvq.metrics import classification_metrics_from_oof
from src.single_view_mglvq.oof import run_oof_predictions
from src.single_view_mglvq.linking import join_structure_and_predictions, perf_vs_structure


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _required_view_files_exist(mdir: Path, do_grid: bool) -> bool:
    required = [
        mdir / "best_params.pkl",
        mdir / "oof_predictions.csv",
    ]
    if do_grid:
        required.append(mdir / "grid.csv")

    return all(p.exists() for p in required)


def _load_existing_view_result(
    *,
    mdir: Path,
    name: str,
    do_grid: bool,
) -> tuple[dict, float, pd.DataFrame, pd.DataFrame | None]:
    best_params = _load_pickle(mdir / "best_params.pkl")
    oof = pd.read_csv(mdir / "oof_predictions.csv")

    grid_df = None
    if do_grid and (mdir / "grid.csv").exists():
        grid_df = pd.read_csv(mdir / "grid.csv")
        best_score = float(grid_df["score"].max())
    else:
        metrics = classification_metrics_from_oof(oof)
        best_score = metrics["balanced_accuracy_from_oof"]

    return best_params, best_score, oof, grid_df


def run_single_view_mglvq(
    *,
    matrices: dict[str, pd.DataFrame],
    y: pd.Series,
    model_cls: type,
    run_name: str,
    out_root: str | Path = "outputs/single_view_mglvq",
    order: str = "y",
    do_grid: bool = True,
    param_grid: dict | None = None,
    fixed_params: dict | None = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    pointwise_structure_df: pd.DataFrame | None = None,
    show_progress: bool = True,
    resume: bool = True,
    force_recompute: bool = False,
) -> dict[str, Any]:
    out_root = Path(out_root)
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    aligned = align_views_and_target(
        matrices=matrices,
        y=y,
        order=order,
    )

    ids_np = aligned.used_ids.astype(str).to_numpy()
    _save_pickle(list(aligned.used_ids.astype(str)), run_dir / "used_ids.pkl")

    rows = []
    joined_outputs = {}

    matrix_items = list(aligned.matrices.items())
    matrix_iterator = matrix_items
    if show_progress:
        matrix_iterator = tqdm(
            matrix_items,
            total=len(matrix_items),
            desc=f"Matrizen {run_name}",
            leave=True,
        )

    for name, D_df in matrix_iterator:
        mdir = run_dir / name
        mdir.mkdir(parents=True, exist_ok=True)

        if show_progress and hasattr(matrix_iterator, "set_postfix"):
            matrix_iterator.set_postfix({"matrix": name})

        should_load_existing = (
            resume
            and not force_recompute
            and _required_view_files_exist(mdir, do_grid=do_grid)
        )

        grid_df = None

        if should_load_existing:
            best_params, best_score, oof, grid_df = _load_existing_view_result(
                mdir=mdir,
                name=name,
                do_grid=do_grid,
            )
        else:
            if do_grid:
                if not param_grid:
                    raise ValueError("param_grid fehlt.")
                best_params, best_score, grid_df = grid_search_best(
                    D_df=D_df,
                    y_arr=aligned.y,
                    ids=ids_np,
                    model_cls=model_cls,
                    param_grid=param_grid,
                    n_splits=n_splits,
                    shuffle=shuffle,
                    random_state=random_state,
                    show_progress=show_progress,
                    progress_desc=f"Grid {run_name} | {name}",
                )
                grid_df.to_csv(mdir / "grid.csv", index=False)
            else:
                if not fixed_params:
                    raise ValueError("fixed_params fehlen.")
                best_params = dict(fixed_params)
                best_score = np.nan

            oof = run_oof_predictions(
                D_df=D_df,
                y_arr=aligned.y,
                ids=ids_np,
                model_cls=model_cls,
                params=best_params,
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
            oof.to_csv(mdir / "oof_predictions.csv", index=False)
            _save_pickle(oof, mdir / "oof_predictions.pkl")
            _save_pickle(best_params, mdir / "best_params.pkl")

        metrics = classification_metrics_from_oof(oof)

        if not do_grid:
            best_score = metrics["balanced_accuracy_from_oof"]

        rows.append({
            "matrix": name,
            "best_score_balanced_accuracy": float(best_score),
            "balanced_accuracy_from_oof": metrics["balanced_accuracy_from_oof"],
            "recall_from_oof": metrics["recall_from_oof"],
            "precision_from_oof": metrics["precision_from_oof"],
            "specificity_from_oof": metrics["specificity_from_oof"],
            "f1_from_oof": metrics["f1_from_oof"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            **{f"best_{k}": v for k, v in best_params.items()},
        })

        # Join-Dateien bei Bedarf neu erzeugen, auch wenn OOF geladen wurde
        if pointwise_structure_df is not None:
            joined = join_structure_and_predictions(
                pointwise_structure_df=pointwise_structure_df,
                oof_df=oof,
                view_name=name,
            )
            joined.to_csv(mdir / "joined_structure_predictions.csv", index=False)

            perf_df = perf_vs_structure(joined)
            perf_df.to_csv(mdir / "perf_vs_structure.csv", index=False)

            joined_outputs[name] = {
                "joined": joined,
                "perf_vs_structure": perf_df,
            }

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    return {
        "run_dir": str(run_dir),
        "used_ids": aligned.used_ids,
        "y_aligned": aligned.y,
        "summary_df": summary_df,
        "joined_outputs": joined_outputs,
    }