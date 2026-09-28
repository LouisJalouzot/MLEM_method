# %%
import pandas as pd
import seaborn as sns

df = pd.read_parquet("experiments/things/fmri/1.parquet")
rename = {col: col.split(".")[-1] for col in df.columns}
df = df.rename(columns=rename)

# %%
df_plot = (
    df.pivot(index=["subject", "roi"], columns="kind", values="mean")
    .reset_index()
    .melt(id_vars=["subject", "roi", "mlem"], value_vars=["frrsa", "rf"])
)
g = sns.relplot(data=df_plot, x="value", y="mlem", hue="subject", col="kind", kind="scatter")
for ax in g.axes.flatten():
    ax.axline((0, 0), slope=1, color="black", linestyle="--")
    ax.set_aspect("equal", adjustable="box")
