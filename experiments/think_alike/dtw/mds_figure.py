# MDS projection of the per-fold model distances for one condition (MLEM and RSA).
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import MDS

from mlem_method.viz import (
    FIGURE_DIR,
    load_cohort,
    load_distance_folds,
    markers,
    mpl,
    palettes,
    plt,
    remove_unused_categories,
    sns,
)

parser = ArgumentParser()
parser.add_argument("--long-range", action="store_true")
args = parser.parse_args()
condition = "long_range" if args.long_range else "rc"
OUTPUT_DIR = FIGURE_DIR / "dtw" / condition

_, index, _, _ = load_cohort()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
folds_by_method = load_distance_folds(condition)

for method in ("mlem", "rsa"):
    folds = folds_by_method[method]
    ref = MDS(metric="precomputed", init="random", random_state=0).fit_transform(sum(folds) / len(folds))
    ref -= ref.mean(0)
    all_coords = []
    for cv, distance in enumerate(folds):
        coords = MDS(metric="precomputed", init="random", random_state=0).fit_transform(distance)
        coords -= coords.mean(0)
        rotation, _ = orthogonal_procrustes(coords, ref)
        all_coords.append(pd.DataFrame(coords @ rotation, columns=["MDS1", "MDS2"], index=index).assign(cv=cv))
    summary = (
        pd.concat(all_coords)
        .reset_index()
        .groupby(["family", "model"], sort=False, observed=True)
        .agg(
            MDS1=("MDS1", "mean"),
            MDS2=("MDS2", "mean"),
            MDS1_sd=("MDS1", "std"),
            MDS2_sd=("MDS2", "std"),
        )
        .fillna(0)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    families = list(summary.groupby("family", sort=False, observed=True))
    xpad = 0.015 * max(np.ptp(summary["MDS1"]), 1e-3)
    ypad = 0.035 * max(np.ptp(summary["MDS2"]), 1e-3)
    handles = []

    for idx, (family, g) in enumerate(families):
        g = remove_unused_categories(g)
        colors = sns.color_palette(palettes[idx % len(palettes)], n_colors=len(g) + 2)[1:-1]
        marker = markers[idx % len(markers)]
        line_color = colors[-1]

        for row, color in zip(g.itertuples(), colors):
            ax.add_patch(
                mpl.patches.Ellipse(
                    (row.MDS1, row.MDS2),
                    2 * row.MDS1_sd,
                    2 * row.MDS2_sd,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.18,
                )
            )

        ax.plot(g["MDS1"], g["MDS2"], color=line_color, linewidth=1.5, alpha=0.9, zorder=1.5)
        sns.scatterplot(
            g,
            x="MDS1",
            y="MDS2",
            hue="model",
            hue_order=g["model"].tolist(),
            palette=colors,
            marker=marker,
            s=70,
            edgecolor="none",
            legend=False,
            ax=ax,
        )

        first = g.iloc[0]
        ax.scatter(
            first["MDS1"],
            first["MDS2"],
            s=70,
            marker=marker,
            facecolors="white",
            edgecolors=colors[0],
            linewidth=1.2,
            zorder=3,
        )

        if family == "gpt2":
            target, ha, va = g.iloc[-1], "center", "top"
            x, y = target["MDS1"], target["MDS2"] - 1.5 * ypad
        elif family == "opt":
            target, ha, va = g.iloc[-1], "right", "center"
            x, y = target["MDS1"] - 6 * xpad, target["MDS2"]
        elif family == "pythia":
            target, ha, va = g.iloc[1], "right", "center"
            x, y = target["MDS1"] - 3 * xpad, target["MDS2"]
        elif family == "OLMo-2":
            target, ha, va = g.iloc[0], "left", "center"
            x, y = target["MDS1"] + 3 * xpad, target["MDS2"]
        elif family == "Llama-3.2":
            target, ha, va = g.iloc[-1], "right", "center"
            x, y = target["MDS1"] - 2 * xpad, target["MDS2"] + ypad
        elif family == "Ministral-3":
            target, ha, va = g.iloc[0], "center", "top"
            x, y = target["MDS1"], target["MDS2"] - 2.5 * ypad
        elif family == "mamba":
            target, ha, va = g.iloc[len(g) // 2], "right", "center"
            x, y = target["MDS1"] - 3 * xpad, target["MDS2"]
        elif family == "mamba2":
            target, ha, va = g.iloc[1], "center", "bottom"
            x, y = target["MDS1"], target["MDS2"] + 2 * ypad
        else:
            target, ha, va = g.iloc[0], "center", "bottom"
            x, y = target["MDS1"], target["MDS2"] + 1.5 * ypad

        ax.text(x, y, family, color=line_color, fontsize=10, fontweight="bold", ha=ha, va=va)
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                marker=marker,
                linestyle="-",
                color=line_color,
                markerfacecolor=line_color,
                markeredgecolor="none",
                markersize=7,
            )
        )

    labels = [family for family, _ in families]
    half = (len(handles) + 1) // 2
    ordered_handles, ordered_labels = [], []
    for i in range(half):
        ordered_handles.append(handles[i])
        ordered_labels.append(labels[i])
        if i + half < len(handles):
            ordered_handles.append(handles[i + half])
            ordered_labels.append(labels[i + half])

    ax.legend(
        ordered_handles,
        ordered_labels,
        title="Family",
        title_fontproperties={"weight": "bold"},
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=5,
        fontsize=9,
        markerscale=1.2,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    ax.scatter(-0.065, -0.03, s=70, facecolors="white", edgecolors="black", clip_on=False, transform=ax.transAxes)
    ax.text(0, -0.03, "smallest model", transform=ax.transAxes, va="center", ha="left", fontsize=9)
    ax.set(xlabel="MDS 1", ylabel="MDS 2")
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    stem = f"{method}_mds"
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
