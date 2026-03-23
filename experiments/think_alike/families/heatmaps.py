# %%
from mlem.viz import *

script_dir = Path(__file__).parent

i, meta = load_df(script_dir / "0.parquet")
s, _ = load_df(script_dir / "1.parquet")
gb_cols = list(meta.columns)

# %%
df = i[i["model"].isin(to_keep)]
df = df.groupby(["Feature"] + gb_cols, observed=True)["FI"].mean().reset_index()
df = df.pivot(
    index=["family", "model", "layer"], columns="Feature", values="FI"
).sort_index()

sim = pairwise_distances(df, metric="euclidean")
sim = pd.DataFrame(sim, index=df.index, columns=df.index)

# %%
src, labels = [], []
family_ticks_x, family_ticks_y, family_names = [], [], []
families = list(df.groupby(level=0, sort=False, observed=True))
gap = 4

for family, g in families:
    block_start = len(src)
    src.extend(g.index.tolist())
    labels.extend([""] * len(g))

    src.extend([None] * gap)
    labels.extend([""] * gap)

    family_ticks_x.append(block_start + (len(g) - 1) / 4)
    family_ticks_y.append(block_start + (len(g) - 1) / 2)
    family_names.append(family)

df_plot = sim.reindex(index=src, columns=src)
df_plot.index = df_plot.columns = labels

# %%
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
mask = np.triu(np.ones_like(df_plot, dtype=bool), k=1)
sns.heatmap(
    df_plot,
    mask=mask,
    cmap="inferno_r",
    vmin=0,
    square=True,
    cbar_kws={
        "shrink": 0.5,
        "orientation": "horizontal",
        "pad": 0.1,
        "label": "Euclidean distance between FI profiles",
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
plt.savefig("think_alike/figures/euclidean_heatmap.pdf", bbox_inches="tight")
