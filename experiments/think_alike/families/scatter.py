# %% Load data
from mlem_method.viz import *

triplets = [
    (
        ("EleutherAI/pythia-6.9b-deduped", 14),
        ("state-spaces/mamba-790m-hf", 21),
        ("Qwen/Qwen3-4B-Base", 16),
        "Relative clause type",
    ),
    (
        ("EleutherAI/pythia-1.4b-deduped", 8),
        ("AntonV/mamba2-780m-hf", 14),
        ("Qwen/Qwen3-4B-Base", 11),
        "Verb lemma",
    ),
]
pairs = [(ref, m, feature) for ref, pos, neg, feature in triplets for m in (pos, neg)]

to_keep = [m for triplet in triplets for m, _ in triplet[:-1]]

script_dir = Path(__file__).parent
mds, _ = load_df(script_dir / "proj.parquet")
mds = mds[mds["method"] == "mds"]
for col, mapping in levels_rename.items():
    mds[col] = mds[col].replace(mapping)
i, meta = load_df(script_dir / "0.parquet", to_keep=to_keep)
gb_cols = list(meta.columns)
max_fi = (
    i[i.set_index(["model_name", "layer"]).index.isin([item for triplet in triplets for item in triplet[:-1]])]
    .groupby(gb_cols + ["Feature"])["FI"]
    .mean()
    .max()
)
top_features = select_top_features(i, n_largest=5)[:5]
other_features = [f for f in i["Feature"].unique() if f not in top_features]
top_features

# %% Plot scatter
n_cols = 2
n_rows = len(pairs) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 4), sharex=True, sharey=True)
tab10 = sns.color_palette("tab10")
feature_to_color = {
    f: tab10[feature_order.index(f) % len(tab10)] if f in feature_order else "0.6" for f in top_features
}
model_labels = meta.drop_duplicates("model_name").set_index("model_name")["model"]

for idx, ((m_ref, l_ref), (m, l), feature) in enumerate(pairs):
    ax = axes[idx // n_cols, idx % n_cols]
    fi_raw = i[(i["model_name"] == m) & (i["layer"] == l)]
    fi = fi_raw.groupby("Feature")["FI"].mean()
    fi_std = fi_raw.groupby("Feature")["FI"].std().reindex(fi.index).fillna(0)
    fi_ref_raw = i[(i["model_name"] == m_ref) & (i["layer"] == l_ref)]
    fi_ref = fi_ref_raw.groupby("Feature")["FI"].mean().loc[fi.index]
    fi_ref_std = fi_ref_raw.groupby("Feature")["FI"].std().reindex(fi.index).fillna(0)

    r2_folds = (
        pd
        .concat(
            [
                fi_raw.groupby(["cv", "Feature"])["FI"].mean().rename("x"),
                fi_ref_raw.groupby(["cv", "Feature"])["FI"].mean().rename("y"),
            ],
            axis=1,
        )
        .dropna()
        .groupby(level="cv")
        .apply(lambda d: d["x"].corr(d["y"]) ** 2)
    )
    r2_mu = r2_folds.mean()
    r2_sd = r2_folds.std()

    ax.scatter(
        x=fi.loc[other_features],
        y=fi_ref.loc[other_features],
        alpha=0.2,
        color="grey",
    )
    ax.plot(
        [-0.05, max_fi * 1.1],
        [-0.05, max_fi * 1.1],
        linestyle="--",
        color="grey",
        linewidth=1,
        alpha=0.6,
    )
    for f in top_features:
        ax.add_patch(
            mpl.patches.Ellipse(
                (fi[f], fi_ref[f]),
                2 * fi_std[f],
                2 * fi_ref_std[f],
                facecolor=feature_to_color[f],
                edgecolor="none",
                alpha=0.12,
            )
        )
        if f == feature:
            color = feature_to_color[f]
            ax.hlines(fi_ref[f], -0.05, fi[f], colors=[color], linewidth=1, alpha=0.8)
            ax.vlines(fi[f], -0.05, fi_ref[f], colors=[color], linewidth=1, alpha=0.8)
    sns.scatterplot(
        x=fi.loc[top_features],
        y=fi_ref.loc[top_features],
        hue=top_features,
        hue_order=top_features,
        palette=feature_to_color,
        edgecolor="none",
        ax=ax,
        legend=False,
    )
    ax.set_aspect("equal")
    ax.set(xlim=(-0.05, max_fi * 1.1), ylim=(-0.05, max_fi * 1.1))
    ax.margins(0)
    ax.text(
        0.05,
        0.95,
        f"R² = {r2_mu:.2f} ± {r2_sd:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="green" if r2_mu > 0.8 else "red",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx % n_cols != 0:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        ax.set_ylabel(f"FI {model_labels[m_ref]} L{l_ref}", fontweight="bold")
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(f"FI {model_labels[m]} L{l}", fontweight="bold")
    ax.xaxis.label.set_visible(True)

axes[0, 0].set_title("Similar", fontweight="bold", pad=15)
axes[0, 1].set_title("Dissimilar", fontweight="bold", pad=15)

handles = [
    mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=5,
        color=feature_to_color[f],
        label=f,
    )
    for f in top_features
] + [
    mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=5,
        color="0.6",
        label="Others",
    )
]

fig.legend(
    handles=handles,
    loc="center",
    bbox_to_anchor=(0.5, 1.1),
    bbox_transform=fig.transFigure,
    ncols=2,
    columnspacing=0.9,
    handletextpad=0.5,
    title="Feature",
    title_fontproperties={"weight": "bold"},
    frameon=True,
)
plt.subplots_adjust(hspace=0.5, wspace=0)
plt.savefig("think_alike/figures/scatter.pdf", metadata={"CreationDate": None})

# %% Plot MDS
n_cols = 3
n_rows = len(triplets)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 6), sharex=False, sharey=False, dpi=150)
for row, ((m_ref, l_ref), (m_pos, l_pos), (m_neg, l_neg), feature) in enumerate(triplets):
    for col, (m, l) in enumerate([(m_ref, l_ref), (m_pos, l_pos), (m_neg, l_neg)]):
        ax = axes[row, col]
        df_plot = mds[(mds["model_name"] == m) & (mds["layer"] == l)]
        # Shuffle for better visualization
        df_plot = df_plot.sample(frac=1, random_state=0)
        sns.scatterplot(
            data=df_plot,
            x="coord_1",
            y="coord_2",
            edgecolor=None,
            hue=feature,
            palette=sns.light_palette(feature_to_color[feature], df_plot[feature].nunique() + 1)[1:],
            ax=ax,
            s=4,
            legend=col == 1,
            rasterized=True,
        )
        if col == 1:
            sns.move_legend(
                ax,
                "lower center" if row == 0 else "upper center",
                bbox_to_anchor=(0.5, 1.4 if row == 0 else 0),
                ncols=10,
                title=feature,
                title_fontproperties={"weight": "bold"},
                frameon=True,
                markerscale=4,
            )
        match col:
            case 0:
                title = f"Reference\n{model_labels[m]}\nlayer {l}"
            case 1:
                title = f"Similar\n{model_labels[m]}\nlayer {l}"
            case 2:
                title = f"Dissimilar\n{model_labels[m]}\nlayer {l}"
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
        for spine in ax.spines.values():
            spine.set_visible(False)
plt.subplots_adjust(hspace=0.25, wspace=0.1)
plt.savefig("think_alike/figures/proj.pdf", metadata={"CreationDate": None})
