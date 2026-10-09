# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from mlem_method.viz import feature_rename
from mlem_method.viz import llm_palette as palette

root = Path(__file__).parent
output = root.parent.parent / "paper" / "figs" / "llm"
output.mkdir(parents=True, exist_ok=True)
methods = {"mlem": "MLEM", "frrsa": "FR-RSA", "rf": "Random Forest"}
layer_ticks = [1, 3, 6, 9, 12]

# Count each fit once, independently of the PFI mode.
scores = pd.read_parquet(root / "1.parquet", filters=[("split", "==", "test"), ("pfi", "==", "feature")])
fi = pd.read_parquet(root / "0.parquet", filters=[("split", "==", "test")])
weights = pd.read_parquet(root / "2.parquet", filters=[("pfi", "==", "feature")])
for df in (scores, fi, weights):
    df.rename(columns=lambda col: col.split(".")[-1], inplace=True)
    df["layer"] = pd.to_numeric(df["layer"])
    df["Method"] = df["kind"].map(methods)

# %% Spearman fit across layers
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
ax.set(xlabel="Layer", ylabel="Spearman $\\rho$ (↑)", xticks=layer_ticks, ylim=(0, 1))
sns.despine(trim=True)
sns.move_legend(ax, bbox_to_anchor=(1, 0.5), loc="center left", title=None, frameon=False)
plt.savefig(output / "spearman.pdf", metadata={"CreationDate": None})

# %% Feature and MLEM term importance
fi["Feature"] = fi["AllFeatures"].map(lambda names: " ×\n".join(feature_rename.get(name, name) for name in names))
feature_fi = fi.query("pfi == 'feature' and Order == 'main'")
term_fi = fi.query("kind == 'mlem' and pfi == 'term'")
features = feature_fi.groupby("Feature")["mean"].mean().nlargest(5).index
# Peak fold-mean FI retains terms prominent only in early layers.
terms = term_fi.groupby(["Feature", "layer"])["mean"].mean().groupby("Feature").max().nlargest(5).index

for data, top, mode, order, aspect in [
    (term_fi, terms, "term", ["MLEM"], 1.5),
    (feature_fi, features, "feature", list(methods.values()), 1),
]:
    g = sns.relplot(
        data=data[data["Feature"].isin(top)],
        x="layer",
        y="mean",
        hue="Feature",
        hue_order=top,
        palette=palette,
        col="Method",
        col_order=order,
        kind="line",
        errorbar="sd",
        marker="o",
        height=2,
        aspect=aspect,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Layer", f"{mode.title()} Importance ($\\Delta\\rho$)")
    g.set(xticks=layer_ticks)
    g.legend.set_title(None)
    sns.despine(trim=True)
    plt.savefig(output / f"{mode}_importance.pdf", metadata={"CreationDate": None})

# %% Training duration
durations = weights[["Method", "layer", "cv", "training_duration"]].drop_duplicates()
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
plt.savefig(output / "training_duration.pdf", metadata={"CreationDate": None})
