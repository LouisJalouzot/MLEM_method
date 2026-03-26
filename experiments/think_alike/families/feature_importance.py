# %% Load data
from mlem.viz import *

i, meta = load_df(Path(__file__).parent / "0.parquet", to_keep=to_keep)

print("Top features:", top_features := select_top_features(i))
df_plot = i[i["Feature"].isin(top_features)]

# %% Feature Importance
g = sns.relplot(
    df_plot,
    x="layer",
    y="FI",
    hue="Feature",
    row="family",
    col="rank",
    kind="line",
    errorbar="sd",
    markers=True,
    style="Feature",
    dashes=False,
    facet_kws={"sharex": False, "sharey": True},
    aspect=1.25,
    height=1.9,
    markersize=4,
    markeredgecolor=None,
)


def customize_facets(data, **kwargs):
    ax = plt.gca()
    model = data["model"].iloc[0]
    title = model.replace("-hf", "")
    ax.set_title("")
    ax.text(
        0.9,
        0.9,
        title,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="top",
        ha="right",
    )


g.map_dataframe(customize_facets)

for ax in g.axes.flat:
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    if not ax.lines:
        ax.set_visible(False)

sns.despine(trim=True)
for ax in g.axes.flat:
    if not ax.get_subplotspec().is_first_col():
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)

    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel("Layer")
    else:
        ax.set_xlabel("")

for line in g.legend.get_lines():
    line.set_linewidth(4)
sns.move_legend(
    g,
    "center right",
    bbox_to_anchor=(0.825, 0.55),
    title="Feature",
    frameon=True,
    title_fontproperties={"weight": "bold"},
    markerscale=4,
    borderpad=1.5,
    fontsize=16,
    title_fontsize=16,
    labelspacing=0.8,
    handletextpad=1,
    handlelength=3,
)
g.fig.subplots_adjust(top=0.975, wspace=0.05, hspace=0.13)
plt.savefig("think_alike/figures/feature_importance.pdf", bbox_inches="tight")
