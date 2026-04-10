# %% Load data
from mlem_method.viz import *

s, meta = load_df(Path(__file__).parent / "1.parquet", to_keep=to_keep)

# %% Spearman
families = meta["family"].unique()
cols = 2
rows = (len(families) + cols - 1) // cols
fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(cols * 4.5, rows * 1.75),
    squeeze=False,
    sharey=True,
)

for idx, (ax, family) in enumerate(zip(axes.flat, families)):
    df_plot = s[s["family"] == family]
    marker = markers[idx % len(markers)]
    palette = sns.color_palette(palettes[idx % len(palettes)], n_colors=df_plot["model"].nunique())

    sns.lineplot(
        remove_unused_categories(df_plot),
        x="rel. layer",
        y="Spearman",
        hue="model",
        errorbar="sd",
        dashes=False,
        marker=marker,
        markersize=4,
        markeredgecolor=None,
        palette=palette,
        ax=ax,
    )
sns.despine(trim=True)

for idx, ax in enumerate(axes.flat):
    if idx < len(families):
        ax.set_title(families[idx].upper(), fontweight="bold", loc="right", fontsize=10, pad=10, y=0.65)
    else:
        ax.set_title("")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))

    if idx % cols != 0:
        ax.get_yaxis().set_visible(False)
        ax.spines["left"].set_visible(False)
    if idx + cols < len(families):
        ax.get_xaxis().set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Relative layer")

    ax.legend(
        *ax.get_legend_handles_labels(),
        loc="center left",
        fontsize=10,
        bbox_to_anchor=(1, 0.5),
        borderpad=1,
        frameon=False,
        title=None,
    )

for ax in axes.flat[len(families) :]:
    ax.set_axis_off()

plt.subplots_adjust(right=0.75, wspace=0.9, hspace=-0.1)
plt.savefig("think_alike/figures/spearman.pdf", bbox_inches="tight")
