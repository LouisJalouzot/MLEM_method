# %% Setup
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import weightedtau

root = Path(__file__).parent
output = root.parent.parent / "paper" / "figs" / "simulation"
output.mkdir(exist_ok=True)

methods = {"mlem": "MLEM", "frrsa": "FR-RSA", "rf": "Random Forest", "oracle": "Oracle"}
hue_order = list(methods.values())[:-1]
metrics = {
    "spearman": "Spearman $\\rho$ (↑)",
    "kendalltau": "FI weighted Kendall $\\tau_w$\nwith Oracle (↑)",
    "euclidean": "FI Euclidean distance\nto Oracle (↓)",
}
col_order = list(metrics.values())
gb_cols = ["Method", "q", "n", "noise", "seed"]


def count_cards(cards: str) -> int:
    cards = literal_eval(cards)
    return sum(cards) - len(cards)


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(root / path, filters=[("split", "==", "test")])
    for col in df:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    rename = {col: col.split(".")[-1] for col in df.columns}
    df = df.rename(columns=rename)

    cardinalities = df["category_cardinalities"]
    cardinalities = cardinalities.map({x: count_cards(x) for x in cardinalities.unique()})
    df["q"] = cardinalities + df["n_numeric"]
    df["Method"] = df["kind"].map(methods)

    return df


df = load("0.parquet").query("Order == 'main'")[["mean", "Feature"] + gb_cols].rename(columns={"mean": "fi"})
df_oracle = df[df["Method"] == "Oracle"].rename(columns={"fi": "fi_oracle"}).drop(columns=["Method"])
df = df[df["Method"] != "Oracle"].merge(df_oracle)
df_score = load("1.parquet")[["mean"] + gb_cols].rename(columns={"mean": "spearman"})
df_score = df_score[df_score["Method"] != "Oracle"]

df = (
    df.groupby(gb_cols, group_keys=False)
    .apply(
        lambda group: pd.Series(
            {
                "euclidean": np.linalg.norm(group.fi - group.fi_oracle),
                "kendalltau": weightedtau(group.fi, group.fi_oracle).statistic,
            }
        )
    )
    .reset_index()
)

df = df.merge(df_score).melt(
    id_vars=gb_cols,
    value_vars=["spearman", "euclidean", "kendalltau"],
    var_name="Metric",
    value_name="value",
)
df["Metric"] = df["Metric"].map(metrics)
df

# %% Noise robustness
df_plot = df[(df.n == 512) & (df.q == 56)]
print(df_plot.groupby(["Method", "Metric", "noise"]).count())
g = sns.relplot(
    data=df_plot,
    x="noise",
    y="value",
    hue="Method",
    hue_order=hue_order,
    kind="line",
    errorbar="sd",
    marker="o",
    col="Metric",
    col_order=col_order,
    facet_kws={"sharey": False},
    height=1.75,
    aspect=1.25,
)
for ax in g.axes.flat:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

g.set_titles("{col_name}")
g.set_axis_labels("", "")
g.axes.flatten()[1].set_xlabel("Noise level $\\sigma$ ($n=512$, $q=56$)")
sns.move_legend(g, "lower center", bbox_to_anchor=(0.475, 0.95), ncols=3, title=None)
plt.savefig(output / "noise_robustness.pdf", bbox_inches="tight", metadata={"CreationDate": None})

# %% Sample efficiency
df_plot = df[(df.noise == 1.0) & (df.q == 56)]
print(df_plot.groupby(["Method", "Metric", "n"]).count())
g = sns.relplot(
    data=df_plot,
    x="n",
    y="value",
    hue="Method",
    hue_order=hue_order,
    kind="line",
    errorbar="sd",
    marker="o",
    col="Metric",
    col_order=col_order,
    facet_kws={"sharey": False},
    height=1.75,
    aspect=1.25,
)
for ax in g.axes.flat:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):d}"))

g.set_titles("{col_name}")
g.set_axis_labels("", "")
g.axes.flatten()[1].set_xlabel("Sample size $n$ (log scale, $q=56$, $\\sigma=1.0$)")
sns.move_legend(g, "lower center", bbox_to_anchor=(0.475, 0.95), ncols=3, title=None)
plt.savefig(output / "sample_efficiency.pdf", bbox_inches="tight", metadata={"CreationDate": None})

# %% Cardinality efficiency
df_plot = df[(df.noise == 1.0) & (df.n == 512)]
print(df_plot.groupby(["Method", "Metric", "q"]).count())
g = sns.relplot(
    data=df_plot,
    x="q",
    y="value",
    hue="Method",
    hue_order=hue_order,
    kind="line",
    errorbar="sd",
    marker="o",
    col="Metric",
    col_order=col_order,
    facet_kws={"sharey": False},
    height=1.75,
    aspect=1.25,
)
for ax in g.axes.flat:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):d}"))

g.set_titles("{col_name}")
g.set_axis_labels("", "")
g.axes.flatten()[1].set_xlabel("Number of feature coordinates $q$ (log scale, $n=512$, $\\sigma=1.0$)")
sns.move_legend(g, "lower center", bbox_to_anchor=(0.475, 0.95), ncols=3, title=None)
plt.savefig(output / "cardinality_efficiency.pdf", bbox_inches="tight", metadata={"CreationDate": None})
