from pathlib import Path
from typing import Dict, Any
import pickle

import pandas as pd

from src.common.alignment import align_views_and_target
from src.structural import single_view as sv
from src.structural import cross_view as cv


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_structural_analysis(
    *,
    matrices: dict[str, pd.DataFrame],
    y: pd.Series,
    run_name: str,
    out_root: str | Path = "outputs/structural",
    order: str = "y",
    k: int = 10,
    n_perm: int = 50,
    seed: int = 0,
    corr_methods: tuple[str, ...] = ("pearson", "spearman"),
    knn_k: int = 10,
    agree_quantile: float = 0.10,
) -> Dict[str, Any]:
    out_root = Path(out_root)
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    aligned = align_views_and_target(
        matrices=matrices,
        y=y,
        order=order,
    )

    _save_pickle(list(aligned.used_ids.astype(str)), run_dir / "used_ids.pkl")

    # ---------------------------------
    # 1) Single-view structural analysis
    # ---------------------------------
    single_rows = {}
    point_tables = {}

    for name, D_df in aligned.matrices.items():
        res = sv.run_single_matrix_analysis(
            D_df=D_df,
            y=aligned.y,
            k=k,
            n_perm=n_perm,
            seed=seed,
            matrix_name=name,
            out_dir=run_dir,
            save_point_table=True,
        )

        summary_df = sv.results_to_summary_df(name, res)
        single_rows[name] = summary_df.reset_index().iloc[0].to_dict()

        if "point_table" in res:
            pt = res["point_table"].copy()
            pt["view"] = name
            point_tables[name] = pt

    single_summary_df = pd.DataFrame(single_rows.values())
    single_summary_df.to_csv(run_dir / "single_view_summary.csv", index=False)

    if point_tables:
        pointwise_df = pd.concat(point_tables.values(), axis=0).reset_index()
        pointwise_df = pointwise_df.rename(columns={"index": "customer_id"})
        pointwise_df.to_csv(run_dir / "pointwise_structure.csv", index=False)
    else:
        pointwise_df = pd.DataFrame()

    # ---------------------------------
    # 2) Cross-view structural comparison
    # ---------------------------------
    cross_dir = run_dir / "cross_view"
    cross_dir.mkdir(parents=True, exist_ok=True)

    corr_results = {}
    corr_summary_results = {}

    for method in corr_methods:
        corr_df = cv.pairwise_matrix_correlation(
            aligned.matrices,
            method=method,
            ids=aligned.used_ids,
        )
        corr_df.to_csv(cross_dir / f"correlation_{method}.csv")

        corr_summary_df = cv.correlation_summary(corr_df)
        corr_summary_df.to_csv(cross_dir / f"correlation_{method}_summary.csv")

        corr_results[method] = corr_df
        corr_summary_results[method] = corr_summary_df

    overlap_mean_df, overlap_dist, used_ids = cv.pairwise_knn_overlap(
        aligned.matrices,
        k=knn_k,
        ids=aligned.used_ids,
    )
    overlap_mean_df.to_csv(cross_dir / f"knn_overlap_k{knn_k}.csv")
    cv.overlap_summary(overlap_mean_df).to_csv(cross_dir / f"knn_overlap_k{knn_k}_summary.csv")
    _save_pickle(overlap_dist, cross_dir / f"knn_overlap_k{knn_k}_distributions.pkl")

    y_series_used = pd.Series(aligned.y, index=aligned.used_ids.astype(str), name="y")
    agreement_df, thresholds, _ = cv.pairwise_agreement_disagreement(
        aligned.matrices,
        quantile=agree_quantile,
        ids=aligned.used_ids,
        y=y_series_used,
    )
    agreement_df.to_csv(cross_dir / f"agreement_q{agree_quantile:.2f}.csv")
    _save_pickle(thresholds, cross_dir / f"agreement_thresholds_q{agree_quantile:.2f}.pkl")
    cv.disagreement_score(agreement_df, subset="all").to_csv(
        cross_dir / f"disagreement_score_q{agree_quantile:.2f}_all.csv"
    )

    return {
        "run_dir": str(run_dir),
        "used_ids": aligned.used_ids,
        "y_aligned": aligned.y,
        "single_summary_df": single_summary_df,
        "pointwise_structure_df": pointwise_df,
        "correlation_results": corr_results,
        "correlation_summary_results": corr_summary_results,
        "overlap_mean_df": overlap_mean_df,
        "agreement_df": agreement_df,
        "agreement_thresholds": thresholds,
    }
