# %% Load data
# uv pip install "git+https://github.com/LouisJalouzot/MLEM.git"
from argparse import ArgumentParser

import torch
from mlem.mlem import MLEM
from scipy.linalg import orthogonal_procrustes

from mlem_method.viz import *

parser = ArgumentParser()
parser.add_argument("--input", type=Path, default=Path(__file__).parent / "0.parquet")
parser.add_argument("--output-dir", type=Path, default=Path("think_alike/figures/dtw"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

i, meta = load_df(args.input, to_keep=to_keep)
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
                dtw_dfs[cv].loc[(family_2, model_2), (family_1, model_1)] = dtw(x, y).normalizedDistance
                pbar.update(1)

# %% Model metadata analyses
dtw_sym = [d.combine_first(d.T).fillna(0.0) for d in dtw_dfs]
all_features = metadata.merge(meta[["model_name", "family", "model"]].drop_duplicates())
all_features = all_features.set_index(["family", "model"]).loc[index].reset_index()
features_by_analysis = {
    "preliminary": all_features.drop(columns=["model_name", "family", "model"]),
    "main": all_features[
        [
            "Family",
            "Architecture",
            "Num. Parameters",
            "Depth",
            "Training Tokens",
            "Non-linearity",
            "Tied Embeddings",
        ]
    ],
}
outputs = {
    "preliminary": args.output_dir / "preliminary_correlations.pdf",
    "main": args.output_dir / "main_correlations.pdf",
}
all_fis = {}
for analysis, features in features_by_analysis.items():
    mlem = MLEM(distance="precomputed", random_seed=0, device="cuda" if torch.cuda.is_available() else "cpu")
    X = mlem._encode_df(features)
    features_dist = (X[None] - X[:, None]).abs().clip(0, 1).nan_to_num(0)

    triu_indices = np.triu_indices(features_dist.shape[0], k=1)
    corrs = pd.DataFrame(features_dist[*triu_indices], columns=features.columns).corr().iloc[1:, :-1]
    mask = np.triu(np.ones_like(corrs, dtype=bool), k=1)
    annot_corrs = corrs.round(2).where(corrs.abs() > 0.3, "")
    _, ax = plt.subplots(figsize=(10, 8) if analysis == "preliminary" else (6, 6))
    sns.heatmap(
        corrs,
        cmap="RdBu_r",
        center=0,
        mask=mask,
        annot=annot_corrs,
        fmt="",
        vmax=1,
        vmin=-1,
        cbar_kws={
            "orientation": "horizontal",
            "location": "top",
        },
        square=True,
        ax=ax,
    )
    colorbar = ax.collections[0].colorbar
    ticks = colorbar.get_ticks()
    colorbar.set_ticks(ticks[::2])
    colorbar.set_label("Correlation", labelpad=10)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.xaxis.set_ticks_position("top")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(outputs[analysis], bbox_inches="tight")
    plt.close()

    fis = []
    scores = []
    for cv, d in enumerate(tqdm(dtw_sym, desc=analysis)):
        train = np.mean([other for k, other in enumerate(dtw_sym) if k != cv], axis=0)
        mlem = MLEM(distance="precomputed", random_seed=0, device="cuda" if torch.cuda.is_available() else "cpu")
        mlem.fit(features_dist, train, feature_names=features.columns)
        fi, score = mlem.score(features_dist, d)
        scores.append(score.mean())
        fis.append(fi.melt(var_name="Feature", value_name="Feature Importance").assign(cv=cv))
    print(analysis, np.mean(scores), np.std(scores))
    all_fis[analysis] = pd.concat(fis)

# %% Plot model metadata FI
for analysis, fis in all_fis.items():
    fis = fis.groupby(["cv", "Feature"], as_index=False)["Feature Importance"].mean()
    print(
        analysis, fis.groupby("Feature")["Feature Importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    )
    hue_order = fis.groupby("Feature")["Feature Importance"].mean().sort_values(ascending=False).index
    _, ax = plt.subplots(figsize=(2.25 if analysis == "preliminary" else 1.5, 5 if analysis == "preliminary" else 3.25))
    sns.barplot(
        fis,
        x="Feature Importance",
        y="Feature",
        hue="Feature",
        order=hue_order,
        hue_order=hue_order,
        legend=False,
        orient="h",
        errorbar="sd",
        ax=ax,
    )
    sns.despine(trim=True)
    ax.set_ylabel("")
    plt.subplots_adjust(right=1.1)
    plt.savefig(args.output_dir / f"{analysis}_fi.pdf", bbox_inches="tight")
    plt.close()

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
plt.savefig(args.output_dir / "heatmap.pdf", bbox_inches="tight")


# %% MDS
def fit_mds(d):
    return MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(d)


def align_to_ref(coords, ref):
    coords = coords - coords.mean(0)
    rotation, _ = orthogonal_procrustes(coords, ref)
    return coords @ rotation


dtw_sym = [d.combine_first(d.T).fillna(0.0) for d in dtw_dfs]
ref = fit_mds(sum(dtw_sym) / len(dtw_sym))
ref = ref - ref.mean(0)
all_coords = [
    pd.DataFrame(align_to_ref(fit_mds(d), ref), columns=["MDS1", "MDS2"], index=d.index).assign(cv=cv)
    for cv, d in zip(cvs, dtw_sym)
]
summary = (
    pd.concat(all_coords)
    .reset_index()
    .groupby(["family", "model"], sort=False, observed=True)
    .agg(
        MDS1=("MDS1", "mean"),
        MDS2=("MDS2", "mean"),
        MDS1_sd=("MDS1", "std"),
        MDS2_sd=("MDS2", "std"),
    )
    .fillna(0)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7, 5))
families = list(summary.groupby("family", sort=False, observed=True))
xpad = 0.015 * max(np.ptp(summary["MDS1"]), 1e-3)
ypad = 0.035 * max(np.ptp(summary["MDS2"]), 1e-3)
handles = []

for idx, (family, g) in enumerate(families):
    g = remove_unused_categories(g)
    colors = sns.color_palette(palettes[idx % len(palettes)], n_colors=len(g) + 2)[1:-1]
    marker = markers[idx % len(markers)]
    line_color = colors[-1]

    for row, color in zip(g.itertuples(), colors):
        ax.add_patch(
            mpl.patches.Ellipse(
                (row.MDS1, row.MDS2),
                2 * row.MDS1_sd,
                2 * row.MDS2_sd,
                facecolor=color,
                edgecolor="none",
                alpha=0.18,
            )
        )

    ax.plot(g["MDS1"], g["MDS2"], color=line_color, linewidth=1.5, alpha=0.9, zorder=1.5)

    sns.scatterplot(
        g,
        x="MDS1",
        y="MDS2",
        hue="model",
        hue_order=g["model"].tolist(),
        palette=colors,
        marker=marker,
        s=70,
        edgecolor="none",
        legend=False,
        ax=ax,
    )

    first = g.iloc[0]
    ax.scatter(
        first["MDS1"],
        first["MDS2"],
        s=70,
        marker=marker,
        facecolors="white",
        edgecolors=colors[0],
        linewidth=1.2,
        zorder=3,
    )

    if family == "gpt2":
        target, ha, va = g.iloc[-1], "center", "top"
        x, y = target["MDS1"], target["MDS2"] - 1.5 * ypad
    elif family == "opt":
        target, ha, va = g.iloc[-1], "right", "center"
        x, y = target["MDS1"] - 6 * xpad, target["MDS2"]
    elif family == "pythia":
        target, ha, va = g.iloc[1], "right", "center"
        x, y = target["MDS1"] - 3 * xpad, target["MDS2"]
    elif family == "OLMo-2":
        target, ha, va = g.iloc[0], "left", "center"
        x, y = target["MDS1"] + 3 * xpad, target["MDS2"]
    elif family == "Llama-3.2":
        target, ha, va = g.iloc[-1], "right", "center"
        x, y = target["MDS1"] - 2 * xpad, target["MDS2"] + ypad
    elif family == "Ministral-3":
        target, ha, va = g.iloc[0], "center", "top"
        x, y = target["MDS1"], target["MDS2"] - 2.5 * ypad
    elif family == "mamba":
        target, ha, va = g.iloc[len(g) // 2], "right", "center"
        x, y = target["MDS1"] - 3 * xpad, target["MDS2"]
    elif family == "mamba2":
        target, ha, va = g.iloc[1], "center", "bottom"
        x, y = target["MDS1"], target["MDS2"] + 2 * ypad
    else:
        target, ha, va = g.iloc[0], "center", "bottom"
        x, y = target["MDS1"], target["MDS2"] + 1.5 * ypad

    ax.text(
        x,
        y,
        family,
        color=line_color,
        fontsize=10,
        fontweight="bold",
        ha=ha,
        va=va,
    )

    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            marker=marker,
            linestyle="-",
            color=line_color,
            markerfacecolor=line_color,
            markeredgecolor="none",
            markersize=7,
        )
    )

labels = [family for family, _ in families]
half = (len(handles) + 1) // 2
ordered_handles, ordered_labels = [], []
for i in range(half):
    ordered_handles.append(handles[i])
    ordered_labels.append(labels[i])
    if i + half < len(handles):
        ordered_handles.append(handles[i + half])
        ordered_labels.append(labels[i + half])

ax.legend(
    ordered_handles,
    ordered_labels,
    title="Family",
    title_fontproperties={"weight": "bold"},
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=5,
    fontsize=9,
    markerscale=1.2,
    columnspacing=0.8,
    handletextpad=0.5,
)
ax.scatter(-0.065, -0.03, s=70, facecolors="white", edgecolors="black", clip_on=False, transform=ax.transAxes)
ax.text(0, -0.03, "smallest model", transform=ax.transAxes, va="center", ha="left", fontsize=9)
ax.set(xlabel="MDS 1", ylabel="MDS 2")
ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.savefig(args.output_dir / "mds.pdf", bbox_inches="tight")
