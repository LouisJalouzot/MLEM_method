from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import weightedtau

root = Path(__file__).parent
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


df = load("0.parquet")[["mean", "Feature"] + gb_cols].rename(columns={"mean": "fi"})
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

g = sns.catplot(
    data=df,
    x="Method",
    y="value",
    order=hue_order,
    hue="Method",
    hue_order=hue_order,
    kind="bar",
    errorbar="sd",
    col="Metric",
    col_order=col_order,
    sharey=False,
    height=1.75,
    aspect=1.25,
)
g.map_dataframe(sns.stripplot, x="Method", y="value", order=hue_order, color="black", jitter=0.15, alpha=0.6, zorder=3)
g.set_titles("{col_name}")
g.set_axis_labels("", "")
plt.savefig(root / "pearson_comparison.pdf", bbox_inches="tight", metadata={"CreationDate": None})
plt.savefig(root / "pearson_comparison.png", dpi=300, bbox_inches="tight")
