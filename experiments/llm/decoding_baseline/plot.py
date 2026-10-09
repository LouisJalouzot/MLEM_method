# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mlem_method.viz import feature_rename, llm_palette as palette

root = Path(__file__).parent
output = root.parent.parent.parent / "paper" / "figs" / "llm"
output.mkdir(parents=True, exist_ok=True)

layer_ticks = [1, 3, 6, 9, 12]
# Get most important features from PFI results
fi = pd.read_parquet(
    root.parent / "0.parquet", filters=[("split", "==", "test"), ("pfi", "==", "feature"), ("Order", "==", "main")]
)
fi["Feature"] = fi["Feature"].map(feature_rename)
features = fi.groupby("Feature")["mean"].mean().nlargest(5).index

# %% Decoding across layers
decoding = pd.read_parquet(root / "0.parquet")
decoding.rename(columns=lambda col: col.split(".")[-1], inplace=True)
decoding["layer"] = pd.to_numeric(decoding["layer"])
decoding["Feature"] = decoding["Feature"].map(feature_rename)
decoding = decoding[decoding["Feature"].isin(features)]
plt.figure(figsize=(2, 1.5))
ax = sns.lineplot(
    data=decoding,
    x="layer",
    y="mean",
    hue="Feature",
    hue_order=features,
    palette=palette,
    errorbar=None,
    marker="o",
)
# The export already summarizes folds; use its SD directly.
for feature, group in decoding.groupby("Feature"):
    group = group.sort_values("layer")
    ax.fill_between(
        group.layer, group["mean"] - group["std"], group["mean"] + group["std"], color=palette[feature], alpha=0.2
    )
ax.axhline(0.5, color="grey", linestyle="--")
ax.text(12, 0.51, "Chance", color="grey", fontsize=8, ha="right", va="bottom")
ax.set(xlabel="Layer", ylabel="ROC AUC (↑)", xticks=layer_ticks, ylim=(0.45, 1.05))
sns.despine(trim=True)
sns.move_legend(ax, bbox_to_anchor=(1, 0.5), loc="center left", title=None, frameon=False)
plt.savefig(output / "decoding.pdf", metadata={"CreationDate": None})
