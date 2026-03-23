from mlem.viz import *

script_dir = Path(__file__).parent

i, meta = load_df(script_dir / "0.parquet")
s, _ = load_df(script_dir / "1.parquet")
gb_cols = list(meta.columns)

df_plot = i[i["family"].isin(["opt", "mamba2"])]
top_features = select_top_features(df_plot, n_largest=8, threshold=0.2)
df_plot = df_plot[df_plot["Feature"].isin(top_features)]
df_plot = remove_unused_categories(df_plot)
top_features

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
    aspect=1,
    height=2.25,
    markersize=4,
    markeredgecolor=None,
)


def customize_facets(data, **kwargs):
    ax = plt.gca()
    model = data["model"].iloc[0]
    title = model.replace("-hf", "")
    ax.set_title("")
    ax.text(
        0.05,
        0.92,
        title,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="left",
    )


g.map_dataframe(customize_facets)

for ax in g.axes.flat:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

sns.despine(trim=True)
for ax in g.axes.flat:
    if not ax.get_subplotspec().is_first_col():
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)
    else:
        ax.set_ylabel("FI")

    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel("Layer")
    else:
        ax.set_xlabel("")

sns.move_legend(
    g,
    "upper center",
    ncol=5,
    title="Feature",
    frameon=True,
    title_fontproperties={"weight": "bold"},
)
g.fig.subplots_adjust(top=0.875, wspace=0.05, hspace=0.13)
plt.savefig("think_alike/figures/feature_importance.pdf", bbox_inches="tight")
