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


# %% Line plots
sns.set_theme(style="ticks")

ratios = np.arange(0.1, 0.6, 0.1)
ratio_data = data[[x in ratio_grid for x in zip(data.n, data.q)]].assign(
    ratio_group=lambda x: x.q_over_n.map(lambda r: ratios[np.abs(ratios - r).argmin()])
)

plots = [
    (data.query("q == 56"), "n", "Stimuli $n$ ($q=56$)", "sample_efficiency.pdf"),
    (ratio_data, "ratio_group", "Encoded feature/sample ratio $q/n$", "ratio_robustness.pdf"),
    (data.query("n == 512"), "q", "Encoded features $q$ ($n=512$)", "q_robustness.pdf"),
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
        errorbar="sd",
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
long = data.melt(
    id_vars=["n", "q", "method"],
    value_vars=metrics,
    var_name="metric",
    value_name="value",
)

g = sns.FacetGrid(
    long,
    row="method",
    row_order=methods,
    col="metric",
    col_order=metrics,
)


def heatmap(data, **_):
    table = data.pivot_table(index="q", columns="n", values="value")
    sns.heatmap(table, cbar=False)


g.map_dataframe(heatmap)
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.set_axis_labels("$n$", "$q$")
g.savefig(output / "heatmaps.pdf", bbox_inches="tight")
plt.close(g.fig)
