# %% Load data
from mlem.viz import *

s, meta = load_df(Path(__file__).parent / "1.parquet", to_keep=to_keep)

# %% Spearman
g = sns.relplot(
    s,
    x="rel. layer",
    y="Spearman",
    hue="rank",
    col="family",
    col_wrap=3,
    kind="line",
    errorbar="sd",
    markers=True,
    style="family",
    dashes=False,
    facet_kws={"sharex": False, "sharey": True},
    aspect=2,
    height=2,
    markersize=4,
    markeredgecolor=None,
)

h, r = g.legend.legend_handles, [t.get_text() for t in g.legend.texts]
g.legend.remove()

g.set_titles("")

for f, ax in g.axes_dict.items():
    m = dict(s[s["family"] == f][["rank", "model"]].astype(str).values)
    ax.legend(
        [x for x, y in zip(h, r) if y in m],
        [m[y] for y in r if y in m],
        title="Model",
        title_fontproperties={"weight": "bold"},
        loc="center left",
        fontsize=10,
        bbox_to_anchor=(1, 0.4),
        borderpad=1,
        frameon=True,
    )

sns.despine(trim=True)

for ax in g.axes.flat:
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    if not ax.get_subplotspec().is_first_col():
        ax.get_yaxis().set_visible(False)
        ax.spines["left"].set_visible(False)
    if ax.get_xlabel():
        ax.set_xlabel("Relative layer")

plt.subplots_adjust(right=0.75, wspace=0.9, hspace=0.14)
plt.savefig("think_alike/figures/spearman.pdf", bbox_inches="tight")
