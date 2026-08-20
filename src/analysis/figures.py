from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_structure_vs_prediction_boxplot(
    joined_df: pd.DataFrame,
    metric: str = "knn_purity",
    title: str | None = None,
    out_path: str | Path | None = None,
    matrix_order: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    figure_width_cm: float = 12,
    figure_height_cm: float = 3,
    plot_width_cm: float = 9,
):
    """
    Boxplot einer Strukturmetrik gegen correct/incorrect pro View.

    Erwartet joined_df im Long-Format mit mindestens:
        customer_id, view, correct, <metric>

    Standardfall:
        metric = "knn_purity"
    """

    if matrix_order is None:
        matrix_order = ["naics", "hs", "am"]

    if label_map is None:
        label_map = {
            "naics": "Industry",
            "hs": "Products",
            "am": "Applications",
        }

    required = {"view", "correct", metric}
    missing = required - set(joined_df.columns)
    if missing:
        raise ValueError(f"joined_df fehlt: {missing}")

    plot_df = joined_df.copy()
    plot_df = plot_df[plot_df["view"].isin(matrix_order)].copy()

    cm = 1 / 2.54
    legend_width_cm = figure_width_cm - plot_width_cm

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(figure_width_cm * cm, figure_height_cm * cm),
        gridspec_kw={"width_ratios": [plot_width_cm, legend_width_cm]},
        constrained_layout=True
    )

    if title is None:
        if metric == "knn_purity":
            title = "Local Neighborhood Purity (kNN)"
        elif metric == "mu":
            title = "Margin-based Local Structure"
        else:
            title = metric

    group_gap = 0.95
    inner_positions = [-0.10, 0.10]
    box_width = 0.16

    colors = {
        1: "#4C78A8",  # Correct
        0: "#F58518",  # Incorrect
    }

    centers = []

    for g, matrix in enumerate(matrix_order):
        base = g * group_gap
        sub = plot_df[plot_df["view"] == matrix]

        series_list = [
            sub.loc[sub["correct"] == 1, metric].dropna(),
            sub.loc[sub["correct"] == 0, metric].dropna(),
        ]

        for s, pos_rel, corr in zip(series_list, inner_positions, [1, 0]):
            pos = base + pos_rel

            bp = ax.boxplot(
                s,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=0.9),
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(colors[corr])
                patch.set_edgecolor("black")

        centers.append(base)

    ax.set_xticks(centers)
    ax.set_xticklabels([label_map.get(m, m) for m in matrix_order])
    ax.set_title(title, pad=3)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.5, pad=1.5)

    legend_elements = [
        Patch(facecolor=colors[1], edgecolor="black", label="Correct"),
        Patch(facecolor=colors[0], edgecolor="black", label="Incorrect"),
    ]

    ax_leg.axis("off")
    ax_leg.legend(
        handles=legend_elements,
        loc="center left",
        frameon=True,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02
        )

    return fig


def load_joined_long_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
