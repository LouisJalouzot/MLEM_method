# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from mlem_method.viz import feature_rename

root = Path(__file__).parent
output = root.parent.parent / "paper" / "figs" / "bert"
output.mkdir(parents=True, exist_ok=True)
methods = {"mlem": "MLEM", "frrsa": "FR-RSA", "rf": "Random Forest"}

# %% Spearman fit across layers
scores = pd.read_parquet(root / "1.parquet", filters=[("split", "==", "test")])
scores = scores.rename(columns={col: col.split(".")[-1] for col in scores.columns})
scores["layer"] = pd.to_numeric(scores["layer"])
scores["Method"] = scores["kind"].map(methods)

plt.figure(figsize=(2, 1.5))
ax = sns.lineplot(
    data=scores,
    x="layer",
    y="mean",
    hue="Method",
    hue_order=list(methods.values()),
    errorbar="sd",
    marker="o",
)
ax.set(xlabel="Layer", ylabel="Test Spearman $\\rho$", xticks=[1, 3, 6, 9, 12], ylim=(0, 1))
ax.legend(title=None)
sns.despine(trim=True)
sns.move_legend(ax, bbox_to_anchor=(1, 0.5), loc="center left", frameon=False)
plt.savefig(output / "spearman.pdf", bbox_inches="tight", metadata={"CreationDate": None})

# %% Feature importance across layers
fi = pd.read_parquet(root / "0.parquet", filters=[("split", "==", "test")])
fi = fi.rename(columns={col: col.split(".")[-1] for col in fi.columns})
fi["layer"] = pd.to_numeric(fi["layer"])
fi["Method"] = fi["kind"].map(methods)
fi["Feature"] = fi["AllFeatures"].map(
    lambda names: " × ".join(feature_rename.get(name, feature_rename.get(f"{name}_0", name)) for name in names)
)
features = (
    fi.groupby(["Method", "Feature"])["mean"]
    .mean()
    .groupby("Method")
    .nlargest(3)
    .index.get_level_values("Feature")
    .unique()
)

g = sns.relplot(
    data=fi[fi["Feature"].isin(features)],
    x="layer",
    y="mean",
    hue="Feature",
    hue_order=features,
    col="Method",
    col_order=list(methods.values()),
    kind="line",
    errorbar="sd",
    marker="o",
    height=2,
    aspect=1,
)
g.set_titles("{col_name}")
g.set_axis_labels("Layer", "Feature Importance")
g.set(xticks=[1, 3, 6, 9, 12])
sns.despine(trim=True)
sns.move_legend(g, bbox_to_anchor=(0.775, 0.5), loc="center left", frameon=True)
plt.savefig(output / "feature_importance.pdf", bbox_inches="tight", metadata={"CreationDate": None})


# %% Training duration
weights = pd.read_parquet(root / "2.parquet")
weights = weights.rename(columns={col: col.split(".")[-1] for col in weights.columns})
durations = weights[["kind", "layer", "cv", "training_duration"]].drop_duplicates()
durations["Method"] = durations["kind"].map(methods)

g = sns.catplot(
    data=durations,
    x="training_duration",
    y="Method",
    order=list(methods.values()),
    hue="Method",
    hue_order=list(methods.values()),
    kind="strip",
    jitter=0.25,
    s=20,
    alpha=0.8,
    legend=False,
    height=2,
    aspect=2,
)
sns.barplot(
    data=durations,
    x="training_duration",
    y="Method",
    order=list(methods.values()),
    hue="Method",
    hue_order=list(methods.values()),
    errorbar="sd",
    alpha=0.6,
    err_kws={"zorder": 3},
    legend=False,
    ax=g.ax,
)
g.set_axis_labels("Training duration (s)", "")
g.ax.set_xscale("log")
g.ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
g.ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
g.ax.xaxis.set_minor_formatter(mticker.NullFormatter())
plt.savefig(output / "training_duration.pdf", bbox_inches="tight", metadata={"CreationDate": None})
