# %%
import pandas as pd
import seaborn as sns

df_fmri = pd.read_parquet("experiments/things/fmri/1.parquet").assign(target="fMRI")
df_meg = pd.read_parquet("experiments/things/meg/1.parquet").assign(target="MEG")
df = pd.concat([df_fmri, df_meg])
rename = {col: col.split(".")[-1] for col in df.columns}
df = df.rename(columns=rename)
df["subject"] = df["target"] + " sub-" + df["subject"].astype(str)

# %%
df_plot = (
    df.pivot_table(
        index=["target", "subject", "t", "roi"],
        columns="kind",
        values="mean",
        aggfunc="mean",
        dropna=False,
    )
    .reset_index()
    .melt(id_vars=["subject", "target", "mlem"], value_vars=["frrsa", "rf"])
    .dropna()
)
g = sns.relplot(
    data=df_plot,
    x="value",
    y="mlem",
    hue="subject",
    col="kind",
    row="target",
    kind="scatter",
    facet_kws={"sharex": False, "sharey": False},
    height=2,
    aspect=1.25,
)
g.set_titles("{row_name}")
for row, axes in enumerate(g.axes):
    for ax in axes:
        ax.axline((0, 0), slope=1, color="black", linestyle="--")
    if row == 1:
        axes[0].set_xlabel("FR-RSA Spearman $\\rho$")
        axes[1].set_xlabel("Random Forest Spearman $\\rho$")
    axes[0].set_ylabel("MLEM Spearman $\\rho$")
sns.despine(trim=True)