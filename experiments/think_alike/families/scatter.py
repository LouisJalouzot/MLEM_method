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

for idx, ((m_ref, l_ref), (m, l)) in enumerate(pairs):
    ax = axes[idx // n_cols, idx % n_cols]
    fi = i[(i["model_name"] == m) & (i["layer"] == l)]
    std = fi.groupby("Feature")["FI"].std()
    fi = fi.groupby("Feature")["FI"].mean()
    fi_ref = i[(i["model_name"] == m_ref) & (i["layer"] == l_ref)]
    std_ref = fi_ref.groupby("Feature")["FI"].std().loc[fi.index]
    fi_ref = fi_ref.groupby("Feature")["FI"].mean().loc[fi.index]

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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx % n_cols != 0:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        m_ref = meta.loc[meta["model_name"] == m_ref, "model"].iloc[0]
        ax.set_ylabel(f"FI {m_ref} L{l_ref}")
    m = meta.loc[meta["model_name"] == m, "model"].iloc[0]
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(f"FI {m} L{l}")
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
]
handles.append(
    mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=5,
        color="0.6",
        label="Other features",
    )
)

fig.legend(
    handles=handles,
    loc="center",
    bbox_to_anchor=(0.5, 0.47),
    bbox_transform=fig.transFigure,
    ncol=3,
    title="Feature",
    title_fontproperties={"weight": "bold"},
    frameon=True,
)
plt.subplots_adjust(hspace=1, wspace=-0.3)
plt.savefig("think_alike/figures/scatter.pdf", bbox_inches="tight")
