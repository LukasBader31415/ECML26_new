from pathlib import Path
import pandas as pd

from src.analysis.load_outputs import (
    load_pointwise_structure,
    load_single_view_oof,
    load_single_view_summary,
    load_m3_fixed_oof_from_pickles,
)
from src.analysis.pointwise_merge import (
    merge_structure_and_single_view,
    build_single_view_wide,
)
from src.analysis.single_view_comparison import (
    build_prediction_dominance_groups,
    build_structure_prediction_alignment,
    summarize_view_complementarity,
)
from src.analysis.multi_view_comparison import (
    build_paper_performance_comparison,
    build_fixed_profile_gain_loss,
)


def run_pointwise_analysis(
    *,
    structural_run_dir,
    single_view_run_dir,
    out_dir,
    structure_metric: str = "knn_purity",
):
    out_dir = Path(out_dir)
    pointwise_dir = out_dir / "pointwise"
    summaries_dir = out_dir / "summaries"
    pointwise_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    structure_df = load_pointwise_structure(structural_run_dir)
    oof_df = load_single_view_oof(single_view_run_dir)

    joined_df = merge_structure_and_single_view(structure_df, oof_df)
    joined_df.to_csv(pointwise_dir / "structure_single_view_joined_long.csv", index=False)

    wide_df = build_single_view_wide(joined_df)
    wide_df.to_csv(pointwise_dir / "single_view_pointwise_comparison_wide.csv", index=False)

    wide_dom_df, dominance_summary_df = build_prediction_dominance_groups(wide_df)
    wide_dom_df.to_csv(pointwise_dir / "single_view_pointwise_with_dominance.csv", index=False)
    dominance_summary_df.to_csv(summaries_dir / "prediction_dominance_groups.csv", index=False)

    alignment_df = build_structure_prediction_alignment(wide_df, structure_metric=structure_metric)
    alignment_df.to_csv(summaries_dir / f"structure_prediction_alignment_{structure_metric}.csv", index=False)

    comp = summarize_view_complementarity(wide_df)
    comp["overall"].to_csv(summaries_dir / "view_complementarity_overall.csv", index=False)
    comp["per_view"].to_csv(summaries_dir / "view_complementarity_per_view.csv", index=False)
    comp["pairwise"].to_csv(summaries_dir / "view_complementarity_pairwise.csv", index=False)

    return {
        "structure_df": structure_df,
        "oof_df": oof_df,
        "joined_df": joined_df,
        "wide_df": wide_df,
        "dominance_summary_df": dominance_summary_df,
        "alignment_df": alignment_df,
        "complementarity": comp,
    }


def run_performance_analysis(
    *,
    single_view_run_dir,
    out_dir,
    single_view_wide_df: pd.DataFrame,
    fixed_summary_df: pd.DataFrame | None = None,
    cluster_core_df: pd.DataFrame | None = None,
    fixed_runs_dir = None,
):
    out_dir = Path(out_dir)
    summaries_dir = out_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    single_view_summary_df = load_single_view_summary(single_view_run_dir)

    perf_df = build_paper_performance_comparison(
        single_view_summary_df=single_view_summary_df,
        summary_fixed_df=fixed_summary_df,
        cluster_core_df=cluster_core_df,
    )
    perf_df.to_csv(summaries_dir / "paper_performance_comparison.csv", index=False)

    gain_loss_df = None
    if fixed_runs_dir is not None:
        fixed_oof_df = load_m3_fixed_oof_from_pickles(fixed_runs_dir)
        gain_loss_df = build_fixed_profile_gain_loss(
            fixed_oof_df=fixed_oof_df,
            single_view_wide_df=single_view_wide_df,
        )
        gain_loss_df.to_csv(summaries_dir / "fixed_profile_gain_loss.csv", index=False)

    return {
        "single_view_summary_df": single_view_summary_df,
        "paper_performance_comparison_df": perf_df,
        "gain_loss_df": gain_loss_df,
    }
