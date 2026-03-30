# %% Load data
from mlem.viz import *

i, meta = load_df(Path(__file__).parent / "0.parquet", to_keep=to_keep)
i = smooth_fi_by_layer(i)
i_pivot = i.pivot_table(
    index=["cv", "family", "model", "layer"],
    columns="Feature",
    values="FI",
)

# %% Compute DTW distance per CV fold
cvs = i_pivot.index.get_level_values("cv").unique().tolist()
index = meta[["family", "model"]].drop_duplicates()
index = pd.MultiIndex.from_frame(index)

dtw_dfs = [
    pd.DataFrame(
        np.full((len(index), len(index)), fill_value=np.nan),
        index=index,
        columns=index,
    )
    for _ in cvs
]
pbar = tqdm(total=len(cvs) * len(index) * (len(index) + 1) // 2, desc="Computing DTW")
with pbar:
    for cv in cvs:
        for k, (family_1, model_1) in enumerate(index):
            for family_2, model_2 in index[k:]:
                x = i_pivot.loc[cv, family_1, model_1].values
                y = i_pivot.loc[cv, family_2, model_2].values
                dtw_dfs[cv].loc[(family_2, model_2), (family_1, model_1)] = dtw(
                    x, y
                ).normalizedDistance
                pbar.update(1)

# %% Heatmap
df = sum(dtw_dfs) / len(dtw_dfs)
families = list(df.groupby(level="family", sort=False, observed=True))
family_names = [family for family, _ in families]
family_sizes = [len(g) for _, g in families]
family_boundaries = np.cumsum(family_sizes)
family_ticks = family_boundaries - np.array(family_sizes) / 2

fig, ax = plt.subplots(figsize=(10, 10))
mask = np.triu(np.ones_like(df, dtype=bool), k=1)
sns.heatmap(
    df,
    mask=mask,
    cmap="inferno_r",
    vmin=0,
    square=True,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={
        "shrink": 0.5,
        "orientation": "horizontal",
        "pad": 0.1,
        "label": "DTW distance between FI profiles across layers",
    },
    ax=ax,
)

n = len(df)
for boundary in family_boundaries[:-1]:
    ax.vlines(boundary, boundary, n, colors="white", linewidth=2)
    ax.hlines(boundary, 0, boundary, colors="white", linewidth=2)

ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="both", which="both", length=0)
ax.set_xticks(family_ticks)
ax.set_xticklabels(family_names, rotation=-45, va="top", ha="left")
ax.set_yticks(family_ticks)
ax.set_yticklabels(family_names, rotation=0)
plt.savefig("think_alike/figures/dtw_heatmap.pdf", bbox_inches="tight")

# %% MDS
