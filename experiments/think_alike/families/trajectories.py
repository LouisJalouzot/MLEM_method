# %% Load data
from mlem.viz import *

script_dir = Path(__file__).parent

i, meta = load_df(
    script_dir / "0.parquet",
    to_keep=to_keep,
)
gb_cols = list(meta.columns)

# %% Smooth FI profiles across layers
sigma = 1
i = i.sort_values("layer")
i["FI"] = i.groupby(["cv", "Feature", "model"], observed=True)["FI"].transform(
    gaussian_filter, sigma=sigma
)
i = i[i["family"] == "pythia"]

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
all_coords = pd.concat(all_coords, ignore_index=True)

# %% Plot trajectories
df_plot = all_coords.copy()
df_other = df_plot[df_plot["family"] != "pythia"]
df_other = df_other.groupby(gb_cols).mean(numeric_only=True).reset_index()
df_plot = df_plot[df_plot["family"] == "pythia"]
if df_other.empty:
    ax = None
else:
    ax = sns.lineplot(
        df_other,
        x="PC1",
        y="PC2",
        units="model",
        estimator=None,
        sort=False,
        color="lightgrey",
        legend=False,
        zorder=0,
    )
ax = pca_lineplot2d(df_plot, palette="rocket_r", ax=ax, band_alpha=0.4)
sns.scatterplot(
    df_plot[df_plot["rel. layer"] == 0]
    .groupby(gb_cols)
    .mean(numeric_only=True)
    .reset_index(),
    x="PC1",
    y="PC2",
    color="white",
    edgecolors="white",
    size=10,
    ax=ax,
    zorder=10,
    legend=False,
)
handles, labels = ax.get_legend_handles_labels()
labels = [
    label.split(",", 1)[0].strip("(' ") if label.startswith("(") and "," in label else label
    for label in labels
]
handles.append(
    mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="black",
        color="black",
        label="first layer",
    )
)
labels.append("first layer")
ax.legend(handles, labels, title="Model", prop={"weight": "bold"})
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
plt.savefig("think_alike/figures/trajectories.pdf", bbox_inches="tight")
