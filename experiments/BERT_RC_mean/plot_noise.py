# %% Setup (run from the repository root)
from pathlib import Path

import pandas as pd
import seaborn as sns

root = Path("experiments/BERT_RC_mean")
labels = {"mlem": "MLEM", "rf": "Random Forest", "frrsa": "FR-RSA"}

# %% Read test results and select the full-training noise sweep
fi = pd.read_parquet(root / "0.parquet", filters=[("split", "==", "test")])
scores = pd.read_parquet(root / "1.parquet", filters=[("split", "==", "test")])
for frame in (fi, scores):
    frame["method"] = frame["trainer.kind"].map(labels)
    frame["n"] = pd.to_numeric(frame["trainer.dataloader_builder.n_train"].replace({"None": "6144"}))
    frame["noise"] = pd.to_numeric(frame["trainer.representations.noise_level"])
fi, scores = fi[fi.n == 6144], scores[scores.n == 6144]
assert scores.groupby(["method", "noise"]).cv.agg(["size", "nunique"]).eq(5).all().all()

# %% Compare raw FI with the noiseless reference from the same method/fold
keys = ["method", "cv", "noise"]
wide = fi.pivot(index=["Feature", "Order"], columns=keys, values="mean")
reference = wide.xs(0, level="noise", axis=1).reindex(columns=wide.columns.droplevel("noise"))
reference.columns = wide.columns
assert wide.notna().all().all() and reference.notna().all().all()
data = scores.set_index(keys)[["mean"]].rename(columns={"mean": "Test Spearman ρ"})
data["FI Kendall τ"] = wide.corrwith(reference, method="kendall")
data["Raw FI Euclidean distance"] = (wide - reference).pow(2).sum().pow(0.5)

# %% Mean ± SD across the five folds
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
long = data.reset_index().melt(id_vars=keys, var_name="metric", value_name="value")
order = [name for name in labels.values() if name in long.method.unique()]
g = sns.relplot(data=long, x="noise", y="value", hue="method", hue_order=order, col="metric", kind="line", errorbar="sd", marker="o", palette="colorblind", height=3.2, aspect=1.15, facet_kws={"sharey": False})
g.set_titles("{col_name}").set_axis_labels("Noise SD multiplier", "")
g.set(xticks=[0, 0.1, 0.5, 1, 2])
g.tick_params(axis="x", labelrotation=45)
sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, title=None)
g.savefig(root / "noise.pdf", bbox_inches="tight")
g.savefig(root / "noise.png", dpi=220, bbox_inches="tight")
