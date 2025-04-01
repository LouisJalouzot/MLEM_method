from captum.attr import FeaturePermutation
from tqdm.auto import tqdm
from statsmodels.stats.descriptivestats import describe
from loguru import logger
import torch
from time import time
import numpy as np
import pandas as pd


def compute_stats(data, alpha=0.01):
    return describe(
        data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha
    ).T


def feature_importance(
    model, dataloader, features, n_perm=30, alpha=0.01, warn_ci=0.01
):
    n = len(features)
    logger.info(
        f"Computing permutation feature importance with {n_perm} permutations for {n*(n-1)//2} feature pairs."
    )
    importances = []
    spearman = []
    start = time()

    pfeatures = features[None] + ", " + features[:, None]
    np.fill_diagonal(pfeatures, features)
    pfeatures = pfeatures[*model.triu_indices]

    with torch.no_grad():
        for i, (X_batch, Y_batch) in tqdm(
            enumerate(dataloader),
            desc="Computing feature importance",
            total=n_perm,
        ):
            t = time()
            X_batch_flat = X_batch[:, None] * X_batch[:, :, None]
            X_batch_flat = X_batch_flat[:, *model.triu_indices]

            def f(x):
                pred = model.flat_forward(x)
                return model.spearman(pred, Y_batch)

            feature_perm = FeaturePermutation(f)
            batch_importances = feature_perm.attribute(X_batch_flat).cpu()
            importances.append(batch_importances.double().numpy().squeeze())
            pred = model(X_batch)
            s = model.spearman(pred, Y_batch).item()
            spearman.append(s)
            logger.debug(
                f"Batch {i:<3} / {n_perm} - Duration: {time() - t:.3g}s - Spearman: {s:<8.3g}"
            )
            if i > n_perm - 1:
                break
    importances = np.stack(importances)
    importances = pd.DataFrame(importances, columns=pfeatures)
    importances = compute_stats(importances, alpha=alpha)
    importances = importances.sort_values("mean", ascending=False)
    importances = importances.reset_index(names="Feature")
    spearman = compute_stats(spearman, alpha=alpha).iloc[0]
    logger.info(
        f"Feature importance computed in {time() - start:.3g}s. Mean Spearman = {spearman["mean"]:.3g} ± {spearman["std"]:.3g}"
    )
    low, high = spearman["lower_ci"], spearman["upper_ci"]
    if high - low > warn_ci:
        logger.warning(
            f"Significant variability between the batches: the {(1 - alpha)*100:.3g}% confidence interval of the Spearman correlation is [{low:.3g}, {high:.3g}] which is larger than the threshold {warn_ci:.3g}."
        )

    return importances, spearman
