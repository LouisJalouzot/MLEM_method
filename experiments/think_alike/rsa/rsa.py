# %% Load data
from mlem_method.viz import *

rsa = load_rsa(Path(__file__).parent / "0.parquet")

# %% Reindex and plot
df = sum(rsa) / len(rsa)
src, labels = [], []
family_ticks_x, family_ticks_y, family_names = [], [], []
families = list(df.groupby(level="family", sort=False, observed=True))
gap = 4

for family, g in families:
    block_start = len(src)
    src.extend(g.index.tolist())
    labels.extend([""] * len(g))

    src.extend([None] * gap)
    labels.extend([""] * gap)

    family_ticks_x.append(block_start + (len(g) - 1) / 2)
    family_ticks_y.append(block_start + (len(g) - 1) / 2)
    family_names.append(family)

df_plot = df.reindex(index=src, columns=src)
df_plot.index = df_plot.columns = labels

fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
mask = np.triu(np.ones_like(df_plot, dtype=bool), k=1)
sns.heatmap(
    df_plot,
    mask=mask,
    cmap="inferno",
    vmin=0,
    square=True,
    cbar_kws={
        "shrink": 0.5,
        "orientation": "horizontal",
        "pad": 0.1,
        "label": "Classical RSA Spearman correlation",
    },
    ax=ax,
    rasterized=True,
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="both", which="both", length=0)
ax.set_xticks(family_ticks_x)
ax.set_xticklabels(family_names, rotation=-45, va="top", ha="left")
ax.set_yticks(family_ticks_y)
ax.set_yticklabels(family_names, rotation=0)
plt.savefig("think_alike/figures/rsa_heatmap.pdf", metadata={"CreationDate": None})

# %% Compute MDS
all_coords = []
for d in rsa:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    coords = mds.fit_transform(np.sqrt(2 * (1 - d)))
    coords = pd.DataFrame(coords, columns=["MDS1", "MDS2"], index=d.index)
    all_coords.append(coords)
all_coords = sum(all_coords) / len(all_coords)
all_coords = all_coords.reset_index()

# %% Plot trajectories
families = all_coords.family.unique()
fig, axes = plt.subplots(1, len(families), figsize=(11, 3), sharex=True, sharey=True)
palettes = {"gpt2": "Reds", "opt": "Greens", "pythia": "Blues"}
markers = {"gpt2": "o", "opt": "s", "pythia": "D"}

for i, (ax, family) in enumerate(zip(axes, families)):
    df = all_coords[all_coords.family == family]
    sns.lineplot(
        remove_unused_categories(df),
        x="MDS1",
        y="MDS2",
        hue="model",
        sort=False,
        marker=markers[family],
        palette=palettes[family],
        markeredgecolor=None,
        ax=ax,
    )

    # White for first layer
    sns.scatterplot(
        df[df["rel. layer"] == 0],
        x="MDS1",
        y="MDS2",
        color="white",
        s=16,
        zorder=100,
        legend=False,
        ax=ax,
        marker=markers[family],
    )

    handles, labels = ax.get_legend_handles_labels()
    labels = [label.split("-")[-1] for label in labels]
    labels.append("first layer")
    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            marker=markers[family],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            # markersize=5,
        )
    )
    ax.legend(
        handles,
        labels,
        title=family.upper(),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title_fontproperties={"weight": "bold", "size": "large"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("MDS1", visible=True)
    ax.set_aspect("equal")
    if i > 0:
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.subplots_adjust(wspace=0.6)
plt.savefig("think_alike/figures/rsa_trajectories.pdf", metadata={"CreationDate": None})
