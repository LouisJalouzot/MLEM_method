"""Plot the natural-feature simulation sweeps."""

# %% Setup
from ast import literal_eval
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, t

root = Path(__file__).parent
output = root / "results"
output.mkdir(exist_ok=True)

params = {"diagonal": "Diagonal", "sym": "Symmetric", "cholesky": "Cholesky"}
methods = [*params.values(), "Random forest"]
colors = dict(zip(methods, ["#4C78A8", "#72B7B2", "#E45756", "#F2CF5B"]))
metrics = [
    ("mean", "Spearman $\\rho$ (↑)"),
    ("fi_tau", "FI Kendall $\\tau$ with Oracle (↑)"),
    ("fi_distance", "FI distance to Oracle (↓)"),
]
ratio_grid = {
    (128, 14), (128, 28), (128, 42), (128, 56), (128, 70),
    (256, 28), (256, 56), (256, 84), (256, 98), (256, 126),
    (512, 56), (512, 98), (512, 154), (512, 210), (512, 252),
    (1024, 98), (1024, 210), (1024, 308), (1024, 406), (1024, 518),
}
fi_columns = [
    "Feature", "Order", "mean", "split", "dataset.seed",
    "dataset.simulation.n_numeric", "dataset.simulation.category_cardinalities",
]
geometry_columns = [
    "mean", "split", "dataset.seed", "dataset.simulation.n",
    "dataset.simulation.n_numeric", "dataset.simulation.category_cardinalities",
]

# %% Load data and compare feature importance with Oracle
oracle = pd.read_parquet(
    root / "oracle" / "0.parquet",
    columns=fi_columns,
    filters=[("split", "==", "test")],
)
estimates, geometries = [], []
for folder in ("mlem", "rf"):
    paths = [root / folder]
    extension = root / "mlp" / "efficiency" / "extension" / "rf"
    if folder == "rf" and extension.exists():
        paths.append(extension)

    extra = ["dataset.simulation.n"]
    if folder == "mlem":
        extra.append("trainer.model_builder.param")
    importance = pd.concat(
        [pd.read_parquet(path / "0.parquet", columns=fi_columns + extra, filters=[("split", "==", "test")]) for path in paths],
        ignore_index=True,
    )
    extra = ["trainer.model_builder.param"] if folder == "mlem" else []
    geometry = pd.concat(
        [pd.read_parquet(path / "1.parquet", columns=geometry_columns + extra, filters=[("split", "==", "test")]) for path in paths],
        ignore_index=True,
    )

    if folder == "mlem":
        importance["method"] = importance["trainer.model_builder.param"].map(params)
        geometry["method"] = geometry["trainer.model_builder.param"].map(params)
    else:
        importance["method"] = geometry["method"] = "Random forest"
    estimates.append(importance)
    geometries.append(geometry)

for frame in (oracle, *estimates, *geometries):
    schemas = frame["dataset.simulation.category_cardinalities"]
    cardinalities = {schema: literal_eval(schema) for schema in schemas.unique()}
    numeric = pd.to_numeric(frame["dataset.simulation.n_numeric"])
    p = numeric + schemas.map({schema: len(values) for schema, values in cardinalities.items()})
    frame["q"] = numeric + schemas.map({schema: sum(value - 1 for value in values) for schema, values in cardinalities.items()})
    frame["schema"] = p.astype(str) + "|" + schemas.astype(str)
    if "dataset.simulation.n" in frame:
        frame["n"] = pd.to_numeric(frame["dataset.simulation.n"])
        frame["q_over_n"] = frame.q / frame.n

oracle = {
    key: frame[["Feature", "Order", "mean"]]
    for key, frame in oracle.groupby(["dataset.seed", "schema"])
}
keys = ["dataset.seed", "n", "q", "q_over_n", "schema", "method"]
rows = []
for key, frame in pd.concat(estimates, ignore_index=True).groupby(keys):
    target = oracle.get((key[0], key[4]))
    if target is None:
        continue
    joined = frame.merge(target, on=["Feature", "Order"], suffixes=("", "_oracle"))
    rows.append({
        **dict(zip(keys, key)),
        "fi_tau": kendalltau(joined["mean"], joined["mean_oracle"]).statistic,
        "fi_distance": np.linalg.norm(joined["mean"] - joined["mean_oracle"]),
    })
data = pd.concat(geometries, ignore_index=True).merge(pd.DataFrame(rows), on=keys)
data = data[data.q <= 308]

# %% Line plots
ratios = np.arange(0.1, 0.6, 0.1)
ratio_data = data[[pair in ratio_grid for pair in zip(data.n, data.q)]].copy()
ratio_data["ratio_group"] = ratio_data.q_over_n.map(lambda value: ratios[np.abs(ratios - value).argmin()])
plots = [
    (data[data.q == 56], "n", "Stimuli $n$ ($q=56$)", "grid_sample_efficiency.png", [128, 256, 512, 1024], True),
    (ratio_data, "ratio_group", "Encoded feature/sample ratio $q/n$", "grid_ratio_robustness.png", ratios, False),
    (data[data.n == 512], "q", "Encoded features $q$ ($n=512$)", "grid_q_robustness.png", None, False),
]
for frame, x, xlabel, filename, ticks, log2 in plots:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
    for ax, (value, ylabel) in zip(axes, metrics):
        stats = frame.groupby([x, "method"])[value].agg(mean="mean", std="std", count="count").reset_index()
        stats["ci"] = stats["std"] / np.sqrt(stats["count"]) * stats["count"].map(lambda n: t.ppf(0.975, n - 1))
        for method in methods:
            values = stats[stats.method == method].sort_values(x)
            xx, yy, ci = values[x].to_numpy(), values["mean"].to_numpy(), values.ci.to_numpy()
            ax.plot(xx, yy, marker="o", color=colors[method], label=method)
            ax.fill_between(xx, yy - ci, yy + ci, color=colors[method], alpha=0.12, linewidth=0)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if ticks is not None:
            ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        if log2:
            ax.set_xscale("log", base=2)
    axes[0].legend(frameon=False)
    fig.savefig(output / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)

# %% Heatmaps
fig, axes = plt.subplots(len(methods), len(metrics), figsize=(12.6, 11.2), constrained_layout=True)
for column, ((value, label), cmap) in enumerate(zip(metrics, ["viridis", "plasma", "cividis"])):
    vmax = data.groupby(["n", "q"])[value].mean().max()
    for row, method in enumerate(methods):
        ax = axes[row, column]
        grid = data[data.method == method].groupby(["q", "n"])[value].mean().unstack("n")
        image = ax.imshow(grid.values, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set(xlabel="$n$", ylabel="$q$", title=f"{method} — {label}" if row == 0 else method)
        ax.title.set_fontsize(9)
    fig.colorbar(image, ax=axes[:, column], orientation="horizontal", location="top", shrink=0.9, pad=0.08, label=label)
fig.savefig(output / "grid_heatmaps.png", dpi=240, bbox_inches="tight")
fig.savefig(output / "grid_heatmaps.pdf")
plt.close(fig)
