# Metadata dendrogram and grouped-correlation heatmap for the main cohort.
from pathlib import Path

import numpy as np
import pandas as pd
from mlem_method.viz import MAIN_GROUPS, load_cohort, plt, sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

OUTPUT_DIR = Path("think_alike/figures/dtw")
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
fig, ax = plt.subplots(figsize=(8.2, 6.2))
sns.heatmap(
    plot_correlations,
    cmap="RdBu_r",
    center=0,
    mask=mask,
    annot=plot_correlations.round(2).where(~mask, ""),
    fmt="",
    vmin=-1,
    vmax=1,
    cbar_kws={"orientation": "horizontal", "location": "top"},
    square=True,
    ax=ax,
)
colorbar = ax.collections[0].colorbar
colorbar.set_label("Largest signed feature-pair correlation", labelpad=8)
colorbar.ax.xaxis.set_label_position("top")
colorbar.ax.xaxis.set_ticks_position("top")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "correlations.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "correlations.png", dpi=220, bbox_inches="tight")
plt.close(fig)

distance = (1 - feature_correlations).clip(lower=0).to_numpy(copy=True)
np.fill_diagonal(distance, 0.0)
tree = linkage(squareform(distance, checks=False), method="single", optimal_ordering=True)
fig, ax = plt.subplots(figsize=(11.5, 8.5))
dendrogram(
    tree,
    labels=features.columns,
    orientation="right",
    color_threshold=0.5,
    above_threshold_color="#4d4d4d",
    leaf_font_size=11,
    ax=ax,
)
ax.set_xlim(1.0, 0.0)
ticks = np.arange(0.0, 1.01, 0.2)
ax.set_xticks(1 - ticks, labels=[f"{value:.1f}" for value in ticks])
ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.2)
ax.set(xlabel="Pearson correlation between feature RDMs", ylabel="", title="Model-metadata feature redundancy")
ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "metadata_dendrogram.pdf", metadata={"CreationDate": None})
fig.savefig(OUTPUT_DIR / "metadata_dendrogram.png", dpi=300)
plt.close(fig)
