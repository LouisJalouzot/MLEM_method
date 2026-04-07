# %% Load data
from mlem_method.viz import *

i, meta = load_df(Path(__file__).parent / "0.parquet", to_keep=to_keep)
i = smooth_fi_by_layer(i)
gb_cols = list(meta.columns)

# %% Compute PCA for each CV
all_coords = []
for cv, df in i.groupby("cv"):
    df_pivot = df.pivot_table("FI", gb_cols, "Feature")
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(df_pivot)
    coords = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df_pivot.index)
    coords = coords.reset_index()
    coords["cv"] = cv
    all_coords.append(coords)
all_coords = pd.concat(all_coords)

# %% Plot trajectories
families = meta["family"].unique()
cols = 3
rows = (len(families) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3), squeeze=False)

for idx, (ax, family) in enumerate(zip(axes.flat, families)):
    df_plot = all_coords[all_coords["family"] == family]
    marker = markers[idx % len(markers)]
    palette = palettes[idx % len(palettes)]

    pca_lineplot2d(df_plot, ax=ax, band_alpha=0.4, marker=marker, palette=palette)

    # Plot first layer markers
    sns.scatterplot(
        df_plot[df_plot["rel. layer"] == 0].groupby(gb_cols).mean(numeric_only=True).reset_index(),
        x="PC1",
        y="PC2",
        marker=marker,
        color="white",
        s=16,
        ax=ax,
        zorder=100,
        legend=False,
    )

    ax.set_title(family, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if idx % cols != 0:
        ax.set_ylabel("")
    if idx + cols < len(families):
        ax.set_xlabel("")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            label="first layer",
        )
    )
    labels.append("first layer")
    ax.legend(
        handles,
        labels,
        title=None,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

# Hide unused subplots
for ax in axes.flat[len(families) :]:
    ax.set_axis_off()

plt.subplots_adjust(wspace=0.7, hspace=0.1)
plt.savefig("think_alike/figures/trajectories.pdf", bbox_inches="tight")
