# Heatmaps of per-fold model distances for one condition (MLEM/DTW and model-level RSA).
from argparse import ArgumentParser

import numpy as np

from mlem_method.viz import FIGURE_DIR, load_distance_folds, plt, sns

parser = ArgumentParser()
parser.add_argument("--long-range", action="store_true")
args = parser.parse_args()
condition = "long_range" if args.long_range else "rc"
OUTPUT_DIR = FIGURE_DIR / "dtw" / condition
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for method, folds in load_distance_folds(condition).items():
    df = sum(folds) / len(folds)
    families = list(df.groupby(level="family", sort=False, observed=True))
    family_names = [family for family, _ in families]
    family_sizes = np.array([len(group) for _, group in families])
    family_boundaries = np.cumsum(family_sizes)
    family_ticks = family_boundaries - family_sizes / 2

    fig, ax = plt.subplots(figsize=(10, 10))
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)
    sns.heatmap(
        df,
        mask=mask,
        cmap="inferno_r",
        vmin=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "shrink": 0.5,
            "orientation": "horizontal",
            "pad": 0.1,
            "label": {
                "mlem": "DTW distance between FI profiles across layers",
                "rsa": "Raw RSA distance between models",
            }[method],
        },
        ax=ax,
    )

    n = len(df)
    for boundary in family_boundaries[:-1]:
        ax.vlines(boundary, boundary, n, colors="white", linewidth=2)
        ax.hlines(boundary, 0, boundary, colors="white", linewidth=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks(family_ticks)
    ax.set_xticklabels(family_names, rotation=-45, va="top", ha="left")
    ax.set_yticks(family_ticks)
    ax.set_yticklabels(family_names, rotation=0)
    fig.savefig(OUTPUT_DIR / f"{method}_heatmap.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    plt.close(fig)
