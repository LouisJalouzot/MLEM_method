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

raw = pd.read_parquet(args.input)
model_columns = [column for column in raw if column.endswith("model_name")]
rsa_input = "spearman" in raw and len(model_columns) == 2

if rsa_input:
    left, right = model_columns
    models = raw[left].drop_duplicates()
    _, meta = clean_df(pd.DataFrame({"model_name": models}), meta_cols=["model_name"])
    index = pd.MultiIndex.from_frame(meta[["family", "model"]])
    raw["cv"] = raw.groupby([left, right]).cumcount()
    cvs = raw["cv"].unique().tolist()
    dtw_dfs = []
    for _, batch in raw.groupby("cv"):
        correlations = batch.pivot(index=left, columns=right, values="spearman").loc[models, models]
        correlations = ((correlations + correlations.T) / 2).clip(-1, 1)
        dtw_dfs.append(pd.DataFrame(np.sqrt(2 * (1 - correlations.to_numpy())), index=index, columns=index))
else:
    i, meta = load_df(args.input, to_keep=to_keep)
    i_pivot = i.pivot_table(
        index=["cv", "family", "model", "layer"],
        columns="Feature",
        values="FI",
    )
    cvs = i_pivot.index.get_level_values("cv").unique().tolist()
    index = pd.MultiIndex.from_frame(meta[["family", "model"]].drop_duplicates())
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


def largest_correlation(left, right):
    correlations = np.corrcoef(left.T, right.T)[: left.shape[1], left.shape[1] :]
    return correlations.flat[np.abs(correlations).argmax()]


def grouped_model_importance(mlem, X, Y, groups, n_permutations=100, seed=0):
    """Jointly permute grouped metadata at the model, rather than model-pair, level."""
    device = mlem.device
    X = X.to(device)
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32, device=device)
    rows, columns = torch.triu_indices(len(X), len(X), offset=1, device=device)
    X_pairs = X[rows, columns]
    Y_pairs = Y[rows, columns]
    model = mlem.model_.to(device)
    baseline = model.spearman(model(X_pairs), Y_pairs).item()
    generator = torch.Generator(device=device).manual_seed(seed)
    importances = {}
    for group, members in groups.items():
        indices = torch.tensor([mlem.feature_names.index(member) for member in members], device=device)
        values = []
        for _ in range(n_permutations):
            order = torch.randperm(len(X), device=device, generator=generator)
            permuted = X_pairs.clone()
            permuted[:, indices] = X[order[rows], order[columns]][:, indices]
            values.append(baseline - model.spearman(model(permuted), Y_pairs).item())
        importances[group] = values
    return pd.DataFrame(importances), pd.Series([baseline] * n_permutations, name="spearman")


# %% Model metadata analyses
dtw_sym = [d.combine_first(d.T).fillna(0.0) for d in dtw_dfs]
all_features = metadata.merge(meta[["model_name", "family", "model"]].drop_duplicates())
all_features = all_features.set_index(["family", "model"]).loc[index].reset_index()
preliminary = all_features.drop(columns=["model_name", "family", "model"])
preliminary = preliminary.loc[:, preliminary.notna().mean() > 0.5]
# Family and Release Date are proxies, Depth / Width is derived, and Tokenizer Type and FFN / Gating Type are redundant here.
main_groups = {
    "Training Procedure": [
        "Weight Provenance",
        "Distillation Objective",
        "Learning Rate Schedule",
        "Warmup Fraction",
        "Training Precision",
    ],
    "Architecture Type": [
        "Architecture",
        "Normalization",
        "Non-linearity",
        "Positional Encoding",
        "Attention Type",
        "Tied Embeddings",
    ],
    "Model Scale and Shape": ["Num. Parameters", "Width", "Depth"],
    "Pretraining Data": ["Training Tokens", "Vocabulary Size", "Language Focus"],
}
features_by_analysis = {
    "preliminary": preliminary,
    "main": all_features[[feature for members in main_groups.values() for feature in members]],
}
groups_by_analysis = {
    "preliminary": {feature: [feature] for feature in preliminary.columns},
    "main": main_groups,
}
outputs = {
    "preliminary": args.output_dir / "preliminary_correlations.pdf",
    "main": args.output_dir / "main_correlations.pdf",
}
all_fis = {}
analyses = ["main"] if rsa_input else ["preliminary", "main"]
for analysis in analyses:
    features = features_by_analysis[analysis]
    groups = groups_by_analysis[analysis]
    features = features.copy()
    present = features.notna()
    categorical = features.select_dtypes(exclude="number").columns
    features[categorical] = features[categorical].fillna("None")
    mlem = MLEM(distance="precomputed", random_seed=0, device="cpu")
    X = mlem._encode_df(features)
    missing = torch.as_tensor(~present.to_numpy(), device=X.device)
    missing_pairs = missing[None] | missing[:, None]
    features_dist = (X[None] - X[:, None]).abs().clip(0, 1).nan_to_num(0).masked_fill(missing_pairs, 0)
    triu_indices = np.triu_indices(features_dist.shape[0], k=1)
    vectors = pd.DataFrame(features_dist[*triu_indices].cpu().numpy(), columns=features.columns)
    correlation_vectors = vectors.mask(missing_pairs[*triu_indices].cpu().numpy())
    if analysis == "main":
        corrs = pd.DataFrame(np.eye(len(groups)), index=groups, columns=groups)
        for left, left_members in groups.items():
            for right, right_members in groups.items():
                if left != right:
                    corrs.loc[left, right] = largest_correlation(
                        vectors[left_members].to_numpy(), vectors[right_members].to_numpy()
                    )
        label = "Largest Correlation"
    else:
        corrs = correlation_vectors.corr()
        label = "Correlation"
    cmap, vmin = "RdBu_r", -1
    mask = np.triu(np.ones_like(corrs, dtype=bool))
    annot_corrs = corrs.round(2).where((corrs.abs() > 0.3) & ~mask, "")
    _, ax = plt.subplots(figsize=(10, 8) if analysis == "preliminary" else (6, 6))
    sns.heatmap(
        corrs,
        cmap=cmap,
        center=0,
        mask=mask,
        annot=annot_corrs,
        fmt="",
        vmax=1,
        vmin=vmin,
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
    colorbar.set_label(label, labelpad=10)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.xaxis.set_ticks_position("top")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(outputs[analysis], metadata={"CreationDate": None})
    plt.close()

    fis = []
    scores = []
    targets = (
        [(0, np.mean(dtw_sym, axis=0), np.mean(dtw_sym, axis=0))]
        if rsa_input
        else [
            (cv, np.mean([other for k, other in enumerate(dtw_sym) if k != cv], axis=0), d)
            for cv, d in enumerate(dtw_sym)
        ]
    )
    for cv, train, test in tqdm(targets, desc=analysis):
        mlem = MLEM(distance="precomputed", random_seed=0, device="cpu")
        mlem.fit(features_dist, train, feature_names=features.columns)
        if analysis == "main":
            fi, score = grouped_model_importance(mlem, features_dist, test, groups, seed=cv)
        else:
            fi, score = mlem.score(features_dist, test)
        scores.append(score.mean())
        fis.append(fi.melt(var_name="Feature", value_name="Feature Importance").assign(cv=cv))
    print(analysis, scores[0] if rsa_input else (np.mean(scores), np.std(scores)))
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
        errorbar=None if rsa_input else "sd",
        ax=ax,
    )
    sns.despine(trim=True)
    ax.set_ylabel("")
    plt.subplots_adjust(right=1.1)
    plt.savefig(args.output_dir / f"{analysis}_fi.pdf", metadata={"CreationDate": None})
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
        "label": "Raw RSA distance between models" if rsa_input else "DTW distance between FI profiles across layers",
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
plt.savefig(args.output_dir / "heatmap.pdf", metadata={"CreationDate": None})


# %% MDS
def fit_mds(d):
    return MDS(metric="precomputed", init="random", random_state=0).fit_transform(d)


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
plt.savefig(args.output_dir / "mds.pdf", metadata={"CreationDate": None})
