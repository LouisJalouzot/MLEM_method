# %% Grouped permutation feature importance for one condition (MLEM and RSA).
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from mlem.mlem import MLEM

from mlem_method.viz import (
    FIGURE_DIR,
    GROUP_COLORS,
    MAIN_GROUPS,
    MEMORY,
    load_cohort,
    load_distance_folds,
    plt,
    sns,
    tqdm,
)

parser = ArgumentParser()
parser.add_argument("--long-range", action="store_true")
args = parser.parse_args()
condition = "long_range" if args.long_range else "rc"
OUTPUT_DIR = FIGURE_DIR / "dtw" / condition

_, _, features, features_dist = load_cohort()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
folds_by_method = load_distance_folds(condition)


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
    return pd.DataFrame(importances), baseline


# Cached MLEM fit + grouped permutation importance for one fold.
# ponytail: features_dist/features are closure globals, not hashed; refresh cache if the cohort changes.
@MEMORY.cache
def fold_importance(train, test, seed):
    mlem = MLEM(distance="precomputed", random_seed=0, device="cuda" if torch.cuda.is_available() else "cpu")
    mlem.fit(features_dist, train, feature_names=features.columns)
    fi, score = grouped_model_importance(mlem, features_dist, test, MAIN_GROUPS, seed=seed)
    return fi.melt(var_name="Feature", value_name="Feature Importance"), score


# %%
for method in ("mlem", "rsa"):
    folds = folds_by_method[method]
    fis = []
    scores = []
    for cv, test in enumerate(tqdm(folds, desc=f"{condition} {method}")):
        train = np.mean([distance for fold, distance in enumerate(folds) if fold != cv], axis=0)
        fi, score = fold_importance(train, test, seed=cv)
        scores.append(score)
        fis.append(fi.assign(cv=cv))
    print(condition, method, (np.mean(scores), np.std(scores)))

    fis = pd.concat(fis).groupby(["cv", "Feature"], as_index=False)["Feature Importance"].mean()
    print(fis.groupby("Feature")["Feature Importance"].agg(["mean", "std"]).sort_values("mean", ascending=False))
    hue_order = fis.groupby("Feature")["Feature Importance"].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(1.5, 3.25))
    sns.barplot(
        fis,
        x="Feature Importance",
        y="Feature",
        hue="Feature",
        order=hue_order,
        hue_order=hue_order,
        palette=GROUP_COLORS,
        legend=False,
        orient="h",
        errorbar="sd",
        ax=ax,
    )
    sns.despine(trim=True)
    ax.set_ylabel("")
    ax.set_xlabel("Feature Importance    ")
    fig.tight_layout()
    stem = f"{method}_group_fi"
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    plt.close(fig)
