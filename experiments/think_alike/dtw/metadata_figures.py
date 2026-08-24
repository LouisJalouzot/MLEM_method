# Metadata dendrogram and grouped-correlation heatmap for the main cohort.

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import squareform

from mlem_method.viz import FIGURE_DIR, GROUP_COLORS, MAIN_GROUPS, load_cohort, plt, sns

OUTPUT_DIR = FIGURE_DIR / "dtw"
model_metadata, _, features, features_dist = load_cohort()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

triu_indices = np.triu_indices(len(features), k=1)
vectors = pd.DataFrame(features_dist[*triu_indices].numpy(), columns=features.columns)
feature_correlations = vectors.corr().fillna(0.0)

group_correlations = pd.DataFrame(np.eye(len(MAIN_GROUPS)), index=MAIN_GROUPS, columns=MAIN_GROUPS)
items = list(MAIN_GROUPS.items())
for group_index, (left, left_members) in enumerate(items):
    for right, right_members in items[group_index + 1 :]:
        block = feature_correlations.loc[left_members, right_members].to_numpy()
        correlation = block.flat[np.abs(block).argmax()]
        group_correlations.loc[left, right] = group_correlations.loc[right, left] = correlation

mask = np.triu(np.ones_like(group_correlations, dtype=bool))[1:, :-1]
plot_correlations = group_correlations.iloc[1:, :-1]
fig, ax = plt.subplots(figsize=(5.75, 4.5))
sns.heatmap(
    plot_correlations,
    cmap="RdBu_r",
    center=0,
    mask=mask,
    annot=plot_correlations.round(2).where(~mask, ""),
    fmt="",
    vmin=-1,
    vmax=1,
    cbar_kws={"orientation": "horizontal", "location": "top", "shrink": 0.6},
    square=True,
    ax=ax,
)
colorbar = ax.collections[0].colorbar
colorbar.set_label("Largest correlation across feature groups", labelpad=8)
colorbar.ax.xaxis.set_label_position("top")
colorbar.ax.xaxis.set_ticks_position("top")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "correlations.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
plt.close(fig)

distance = (1 - feature_correlations).clip(lower=0).to_numpy(copy=True)
np.fill_diagonal(distance, 0.0)
tree = linkage(squareform(distance, checks=False), method="single", optimal_ordering=True)
feature_to_group = {feature: group for group, members in MAIN_GROUPS.items() for feature in members}
fig, ax = plt.subplots(figsize=(7.5, 5.5))
set_link_color_palette(["#4d4d4d"])
dendrogram(
    tree,
    labels=features.columns,
    orientation="right",
    color_threshold=0.5,
    above_threshold_color="#c9c9c9",
    leaf_font_size=11,
    ax=ax,
)
for tick in ax.get_yticklabels():
    tick.set_color(GROUP_COLORS[feature_to_group[tick.get_text()]])
ax.legend(
    handles=[Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in GROUP_COLORS.items()],
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncols=1,
)
ax.set_xlim(1.05, -0.05)
ticks = np.arange(0.0, 1.01, 0.2)
ax.set_xticks(1 - ticks, labels=[f"{value:.1f}" for value in ticks])
ax.invert_xaxis()
ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.2)
ax.set(xlabel="Pearson correlation between feature RDMs", ylabel="")
ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "metadata_dendrogram.pdf", metadata={"CreationDate": None})
plt.close(fig)
