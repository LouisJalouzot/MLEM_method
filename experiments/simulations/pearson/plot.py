# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import weightedtau

root = Path(__file__).parent
output = root.parent.parent.parent / "paper" / "figs" / "simulation"
output.mkdir(parents=True, exist_ok=True)

methods = {"mlem": "MLEM", "frrsa": "FR-RSA", "oracle": "Oracle"}
hue_order = list(methods.values())[:-1]
metrics = {
    "pearson": "Pearson $r$ (↑)",
    "kendalltau": "FI weighted Kendall $\\tau_w$\nto Oracle (↑)",
    "euclidean": "FI Euclidean distance\nto Oracle (↓)",
}
col_order = list(metrics.values())
gb_cols = ["Method", "seed"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(root / path, filters=[("split", "==", "test")])
    for col in df:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    df = df.rename(columns={col: col.split(".")[-1] for col in df.columns})
    df["Method"] = df["kind"].map(methods)
    return df


df = load("0.parquet").query("Order == 'main'")[["mean", "Feature"] + gb_cols].rename(columns={"mean": "fi"})
df_oracle = df[df["Method"] == "Oracle"].rename(columns={"fi": "fi_oracle"}).drop(columns=["Method"])
df = df[df["Method"] != "Oracle"].merge(df_oracle)
df_score = load("1.parquet")[["mean"] + gb_cols].rename(columns={"mean": "pearson"})
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
    value_vars=["pearson", "kendalltau", "euclidean"],
    var_name="Metric",
    value_name="value",
)
df["Metric"] = df["Metric"].map(metrics)

# %%
g = sns.catplot(
    data=df,
    x="Method",
    y="value",
    order=hue_order,
    hue="Method",
    hue_order=hue_order,
    col="Metric",
    col_order=col_order,
    kind="strip",
    jitter=0.2,
    alpha=0.8,
    sharey=False,
    height=2,
    aspect=1,
)
for ax, metric in zip(g.axes.flat, col_order):
    sns.barplot(
        data=df[df["Metric"] == metric],
        x="Method",
        y="value",
        order=hue_order,
        hue="Method",
        hue_order=hue_order,
        errorbar="sd",
        linestyle="none",
        alpha=0.6,
        err_kws={"zorder": 3},
        legend=False,
        ax=ax,
    )
g.set_titles("{col_name}", pad=10)
g.set_axis_labels("", "")
plt.savefig(output / "pearson_comparison.pdf", bbox_inches="tight", metadata={"CreationDate": None})
