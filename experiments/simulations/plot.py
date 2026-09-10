# %% Setup
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

root = Path(__file__).parent
output = root.parent.parent / "paper" / "figs" / "simulation"
output.mkdir(exist_ok=True)

params = {"cholesky": "MLEM", "sym": "FR-RSA-I", "diagonal": "Diagonal"}
methods = [*params.values(), "Random Forest"]
metrics = {
    "mean": "Spearman $\\rho$ (↑)",
    "fi_tau": "FI Kendall $\\tau$ with Oracle (↑)",
    "fi_distance": "FI distance to Oracle (↓)",
}

ratio_grid = {
    (128, 14),
    (128, 28),
    (128, 42),
    (128, 56),
    (128, 70),
    (256, 28),
    (256, 56),
    (256, 84),
    (256, 98),
    (256, 126),
    (512, 56),
    (512, 98),
    (512, 154),
    (512, 210),
    (512, 252),
    (1024, 98),
    (1024, 210),
    (1024, 308),
    (1024, 406),
    (1024, 518),
}


def preprocess(df):
    schemas = df["dataset.simulation.category_cardinalities"]
    cards = schemas.map({x: literal_eval(x) for x in schemas.unique()})
    numeric = pd.to_numeric(df["dataset.simulation.n_numeric"])

    df["q"] = numeric + cards.map(lambda x: sum(v - 1 for v in x))
    df["schema"] = (numeric + cards.map(len)).astype(str) + "|" + schemas

    if "dataset.simulation.n" in df:
        df["n"] = pd.to_numeric(df["dataset.simulation.n"])
        df["q_over_n"] = df.q / df.n

    return df


# %% Load data
oracle = preprocess(pd.read_parquet(root / "oracle" / "0.parquet", filters=[("split", "==", "test")]))
oracle = {key: g[["Feature", "Order", "mean"]] for key, g in oracle.groupby(["dataset.seed", "schema"])}

estimates, geometries = [], []

for folder in ("mlem", "rf"):
    paths = [root / folder]
    extension = root / "mlp" / "efficiency" / "extension" / "rf"
    if folder == "rf" and extension.exists():
        paths += [extension]

    is_mlem = folder == "mlem"
    param = ["trainer.model_builder.param"] if is_mlem else []

    importance = preprocess(
        pd.concat([pd.read_parquet(p / "0.parquet", filters=[("split", "==", "test")]) for p in paths])
    )

    geometry = preprocess(
        pd.concat([pd.read_parquet(p / "1.parquet", filters=[("split", "==", "test")]) for p in paths])
    )

    if is_mlem:
        importance["method"] = importance["trainer.model_builder.param"].map(params)
        geometry["method"] = geometry["trainer.model_builder.param"].map(params)
    else:
        importance["method"] = geometry["method"] = "Random Forest"

    estimates.append(importance)
    geometries.append(geometry)


# %% Feature-importance agreement
keys = ["dataset.seed", "n", "q", "q_over_n", "schema", "method"]

rows = []
for key, g in pd.concat(estimates).groupby(keys):
    target = oracle.get((key[0], key[4]))
    if target is None:
        continue

    g = g.merge(target, on=["Feature", "Order"], suffixes=("", "_oracle"))
    rows.append(
        {
            **dict(zip(keys, key)),
            "fi_tau": kendalltau(g["mean"], g["mean_oracle"]).statistic,
            "fi_distance": np.linalg.norm(g["mean"] - g["mean_oracle"]),
        }
    )

data = pd.concat(geometries).merge(pd.DataFrame(rows), on=keys)


# %% Seed coverage (unique seeds, not feature/geometry rows)
coverage = data.groupby(["method", "q", "n"])["dataset.seed"].nunique().unstack("method", fill_value=0)[methods]
print("\nUnique seeds:")
print(coverage.to_string())

# %% Line plots
sns.set_theme(style="ticks")

ratios = np.arange(0.1, 0.6, 0.1)
ratio_data = data[[x in ratio_grid for x in zip(data.n, data.q)]].assign(
    ratio_group=lambda x: x.q_over_n.map(lambda r: ratios[np.abs(ratios - r).argmin()])
)

plots = [
    (data.query("q == 56"), "n", "Stimuli $n$ ($q=56$)", "sample_efficiency.pdf"),
    (ratio_data, "ratio_group", "Encoded coordinate/sample ratio $q/n$", "ratio_robustness.pdf"),
    (data.query("n == 512"), "q", "Encoded coordinates $q$ ($n=512$)", "q_robustness.pdf"),
]

for df, x, xlabel, filename in plots:
    df = df.melt(
        id_vars=[x, "method"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    g = sns.relplot(
        data=df,
        x=x,
        y="value",
        hue="method",
        hue_order=methods,
        col="metric",
        col_order=metrics,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.5,
        aspect=1,
    )
    g.set_xlabels(xlabel)
    g.set_ylabels("")

    for ax, ylabel in zip(g.axes.flat, metrics.values()):
        ax.set_title(ylabel)
        ax.lines[0].set_zorder(10)
        if x == "n":
            ax.set_xscale("log", base=2)
            ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])

    sns.despine(trim=True)
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, 0.95), ncols=4, title=None)
    g.savefig(output / filename, bbox_inches="tight")
    plt.close(g.fig)


# %% Heatmaps
means = data.groupby(["method", "q", "n"])[list(metrics)].mean()
qs, ns = sorted(data.q.unique()), sorted(data.n.unique())
fig, axes = plt.subplots(len(metrics), len(methods), figsize=(14, 10), sharex=True, sharey=True, layout="constrained")

for row, ((metric, title), cmap) in enumerate(zip(metrics.items(), ["Blues", "Oranges", "Purples_r"])):
    # Share limits across methods within each metric row.
    vmin, vmax = means[metric].min(), means[metric].max()
    tables = means[metric].unstack("method").reindex(columns=methods)
    for col, method in enumerate(methods):
        ax = axes[row, col]
        table = tables[method].unstack("n").reindex(index=qs[::-1], columns=ns)
        sns.heatmap(
            table,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=0.35,
            linecolor="white",
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_facecolor("#eeeeeedb")
        ax.tick_params(axis="both", length=0, labelsize=8, labelrotation=0)
        ax.label_outer()
        ax.set_xlabel("Stimuli $n$" if row == len(metrics) - 1 else "")
        ax.set_ylabel(f"{title}\nEncoded coordinates $q$" if col == 0 else "", fontsize=11)
        if row == 0:
            ax.set_title(method, fontsize=12, pad=12)
    fig.colorbar(ax.collections[0], ax=axes[row, :], fraction=0.025, pad=0.02)

fig.supxlabel("Light gray cells: no observations", fontsize=9, color="0.4")
fig.savefig(output / "heatmaps.pdf", bbox_inches="tight")
plt.close(fig)
