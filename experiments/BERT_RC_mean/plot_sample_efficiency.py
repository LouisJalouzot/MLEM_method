# %% Setup (run from the repository root)
from ast import literal_eval
from pathlib import Path

import pandas as pd
import seaborn as sns

root = Path("experiments/BERT_RC_mean")
labels = {"cholesky": "MLEM", "triu": "FR-RSA-I", "diagonal": "Diagonal", "rf": "Random Forest"}

# %% Read test results and select the noiseless training-size sweep
fi = pd.read_parquet(root / "0.parquet", filters=[("split", "==", "test")])
scores = pd.read_parquet(root / "1.parquet", filters=[("split", "==", "test")])
names = {}
for value in fi.trainer.unique():
    trainer = literal_eval(value)
    names[value] = labels[trainer.get("model_builder", {}).get("param", trainer.get("kind"))]
for frame in (fi, scores):
    frame["method"] = frame.trainer.map(names)
    frame["n"] = pd.to_numeric(frame["trainer.dataloader_builder.n_train"].replace({"None": "6144"}))
    frame["noise"] = pd.to_numeric(frame["trainer.representations.noise_level"])
fi, scores = fi[fi.noise == 0], scores[scores.noise == 0]
assert scores.groupby(["method", "n"]).cv.agg(["size", "nunique"]).eq(5).all().all()

# %% Compare raw FI with the full-training reference from the same method/fold
keys = ["method", "cv", "n"]
wide = fi.pivot(index=["Feature", "Order"], columns=keys, values="mean")
reference = wide.xs(6144, level="n", axis=1).reindex(columns=wide.columns.droplevel("n"))
reference.columns = wide.columns
assert wide.notna().all().all() and reference.notna().all().all()
data = scores.set_index(keys)[["mean"]].rename(columns={"mean": "Test Spearman ρ"})
data["FI Kendall τ"] = wide.corrwith(reference, method="kendall")
data["Raw FI Euclidean distance"] = (wide - reference).pow(2).sum().pow(0.5)

# %% Mean ± SD across the five folds
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
long = data.reset_index().melt(id_vars=keys, var_name="metric", value_name="value")
order = [name for name in labels.values() if name in long.method.unique()]
g = sns.relplot(data=long, x="n", y="value", hue="method", hue_order=order, col="metric", kind="line", errorbar="sd", marker="o", palette="colorblind", height=3.2, aspect=1.15, facet_kws={"sharey": False})
g.set_titles("{col_name}").set_axis_labels("Training stimuli (fixed test fold)", "")
g.set(xscale="log", xticks=[256, 512, 1024, 2048, 4096, 6144], xticklabels=[256, 512, 1024, 2048, 4096, 6144])
sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, title=None)
g.savefig(root / "sample_efficiency.pdf", bbox_inches="tight")
g.savefig(root / "sample_efficiency.png", dpi=220, bbox_inches="tight")
