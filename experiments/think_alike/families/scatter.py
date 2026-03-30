# %% Load data
from mlem.viz import *

pairs = [
    (("facebook/opt-1.3b", 17), ("Qwen/Qwen2.5-7B", 20)),
    (("facebook/opt-1.3b", 17), ("fla-hub/rwkv7-191M-world", 9)),
    (("EleutherAI/pythia-6.9b-deduped", 14), ("state-spaces/mamba-790m-hf", 21)),
    (("EleutherAI/pythia-6.9b-deduped", 14), ("Qwen/Qwen3-4B-Base", 16)),
]

to_keep = [m for pair in pairs for m, _ in pair]

script_dir = Path(__file__).parent
mds, _ = load_df(script_dir / "proj.parquet")
i, meta = load_df(script_dir / "0.parquet", to_keep=to_keep)
gb_cols = list(meta.columns)
max_fi = i.groupby(gb_cols + ["Feature"])["FI"].mean().max() * 1.1
top_features = select_top_features(i, n_largest=5)[:5]
other_features = [f for f in i["Feature"].unique() if f not in top_features]
top_features

# %% Scatter
n_cols = 2
n_rows = len(pairs) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4, 6.0), sharex=True, sharey=True)
feature_to_color = dict(
    zip(top_features, sns.color_palette("tab10", n_colors=len(top_features)))
)
model_labels = meta.drop_duplicates("model_name").set_index("model_name")["model"]

for idx, ((m_ref, l_ref), (m, l)) in enumerate(pairs):
    ax = axes[idx // n_cols, idx % n_cols]
    fi_raw = i[(i["model_name"] == m) & (i["layer"] == l)]
    fi = fi_raw.groupby("Feature")["FI"].mean()
    fi_ref_raw = i[(i["model_name"] == m_ref) & (i["layer"] == l_ref)]
    fi_ref = fi_ref_raw.groupby("Feature")["FI"].mean().loc[fi.index]

    r2_folds = (
        pd.concat(
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
        [0, max_fi],
        [0, max_fi],
        linestyle="--",
        color="gray",
        linewidth=1,
        alpha=0.6,
    )
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
    ax.text(
        0.98,
        0.04,
        f"R² = {r2_mu:.2f} ± {r2_sd:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="green" if r2_mu > 0.8 else "red",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx % n_cols != 0:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        ax.set_ylabel(f"FI {model_labels[m_ref]} L{l_ref}")
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(f"FI {model_labels[m]} L{l}")
    ax.xaxis.label.set_visible(True)

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
        label="Other features",
    )
]

fig.legend(
    handles=handles,
    loc="center",
    bbox_to_anchor=(0.5, 0.47),
    bbox_transform=fig.transFigure,
    ncol=3,
    columnspacing=0.9,
    handletextpad=0.5,
    title="Feature",
    title_fontproperties={"weight": "bold"},
    frameon=True,
)
plt.subplots_adjust(hspace=1, wspace=-0.3)
plt.savefig("think_alike/figures/scatter.pdf", bbox_inches="tight")
