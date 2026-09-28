# %%
import pandas as pd
import seaborn as sns

df_fmri = pd.read_parquet("experiments/things/fmri/1.parquet").assign(target="fMRI")
df_meg = pd.read_parquet("experiments/things/meg/1.parquet").assign(target="MEG")
df = pd.concat([df_fmri, df_meg])
rename = {col: col.split(".")[-1] for col in df.columns}
df = df.rename(columns=rename)

# %%
df_plot = (
    df.pivot(index=["target", "subject", "t", "roi"], columns="kind", values="mean")
    .reset_index()
    .melt(id_vars=["subject", "target", "mlem"], value_vars=["frrsa", "rf"])
)
g = sns.relplot(data=df_plot, x="value", y="mlem", hue="subject", col="kind", row="target", kind="scatter")
for ax in g.axes.flatten():
    ax.axline((0, 0), slope=1, color="black", linestyle="--")
    ax.set_aspect("equal", adjustable="box")
