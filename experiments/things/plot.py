# %%
import matplotlib.pyplot as plt
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
median = df_plot.groupby(["target", "kind"], as_index=False)[["value", "mlem"]].median().assign(subject="Median")
df_plot = pd.concat([df_plot, median], ignore_index=True)
subjects = df_plot["subject"].unique()
palette = dict(zip(subjects, sns.color_palette("tab10", len(subjects))))
palette["Median"] = "black"
markers = dict.fromkeys(subjects, "o")
markers["Median"] = "D"
sizes = dict.fromkeys(subjects, 25)
sizes["Median"] = 50
g = sns.relplot(
    data=df_plot,
    x="value",
    y="mlem",
    hue="subject",
    style="subject",
    palette=palette,
    markers=markers,
    size="subject",
    sizes=sizes,
    col="kind",
    row="target",
    kind="scatter",
    facet_kws={"sharex": False, "sharey": False, "gridspec_kws": {"hspace": 0.25}},
    height=2,
    aspect=1.25,
)
g.set_titles("")
for row, axes in enumerate(g.axes):
    for ax in axes:
        ax.text(
            0.1,
            0.9,
            g.row_names[row],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        ax.axline((0, 0), slope=1, color="black", linestyle="--")
    if row == 1:
        axes[0].set_xlabel("FR-RSA Spearman $\\rho$")
        axes[1].set_xlabel("Random Forest Spearman $\\rho$")
    axes[0].set_ylabel("MLEM Spearman $\\rho$")
sns.despine(trim=True)
sns.move_legend(g, "center left", bbox_to_anchor=(0.8, 0.5), title="Subject", frameon=True)
plt.savefig("paper/figs/things.pdf", bbox_inches="tight", metadata={"CreationDate": None})
