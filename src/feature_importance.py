import typing as tp
from time import time

import numpy as np
import pandas as pd
import torch
from captum.attr import FeaturePermutation
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict
from statsmodels.stats.descriptivestats import describe
from torch import nn
from torch.utils import data
from tqdm.auto import tqdm

from src.trainer import Trainer
from src.utils import BaseModel


def compute_stats(data, alpha=0.01):
    """Compute descriptive statistics with confidence intervals"""
    return describe(
        data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha
    ).T


def compute_feature_importance(
    model: nn.Module,
    dataset: data.Dataset,
    features: list[str],
    n_perm: int,
    alpha: float,
    warn_ci: float,
) -> tp.Tuple[pd.DataFrame, pd.Series]:
    """Core logic for computing permutation feature importance."""
    n = len(features)
    logger.info(
        f"Computing permutation feature importance with {n_perm} permutations "
        f"for {n*(n-1)//2} feature pairs."
    )

    # Create feature pair names
    features_arr = np.array(features)
    pfeatures = features_arr[None] + ", " + features_arr[:, None]
    np.fill_diagonal(pfeatures, features)
    pfeatures = pfeatures[*model.triu_indices]

    # Storage for results
    importances = []
    spearman = []
    start = time()

    # Process batches
    with torch.no_grad():
        for i in tqdm(
            range(n_perm),
            desc="Computing feature importance",
        ):
            t = time()

            # Sample a batch
            X_batch, Y_batch = dataset[i]

            # Create flattened feature interactions
            X_batch_flat = X_batch[:, None] * X_batch[:, :, None]
            X_batch_flat = X_batch_flat[:, *model.triu_indices]

            # Define the function to measure attributions
            def f(x):
                pred = model.flat_forward(x)
                return model.spearman(pred, Y_batch)

            # Calculate feature importance
            feature_perm = FeaturePermutation(f)
            batch_importances = feature_perm.attribute(X_batch_flat).cpu()
            importances.append(batch_importances.double().numpy().squeeze())

            # Calculate baseline performance
            pred = model(X_batch)
            s = model.spearman(pred, Y_batch).item()
            spearman.append(s)

            logger.debug(
                f"Batch {i:<3} / {n_perm} - "
                f"Duration: {time() - t:.3g}s - "
                f"Spearman: {s:<8.3g}"
            )

    # Compute importances statistics
    importances = np.stack(importances)
    importances = pd.DataFrame(importances, columns=pfeatures)
    importances_stats = compute_stats(importances, alpha=alpha)
    importances_stats = importances_stats.sort_values("mean", ascending=False)
    importances_stats = importances_stats.reset_index(names="Feature")

    # Compute spearman statistics
    spearman_stats = compute_stats(spearman, alpha=alpha).iloc[0]

    # Log results
    logger.info(
        f"Feature importance computed in {time() - start:.3g}s. "
        f"Mean Spearman = {spearman_stats['mean']:.3g} ± {spearman_stats['std']:.3g}"
    )

    # Warn if there's significant variability on the Spearman correlation
    # across batches
    low, high = spearman_stats["lower_ci"], spearman_stats["upper_ci"]
    if high - low > warn_ci:
        logger.warning(
            f"Significant variability between batches: "
            f"the {(1 - alpha)*100:.3g}% confidence interval "
            f"of the Spearman correlation is [{low:.3g}, {high:.3g}] "
            f"which is larger than the threshold {warn_ci:.3g}."
        )

    return importances_stats, spearman_stats


class FeatureImportance(BaseModel):
    trainer: Trainer

    n_perm: int = 30
    alpha: float = 0.01
    warn_ci: float = 0.01

    infra: TaskInfra = TaskInfra(version="1", folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @infra.apply
    def compute(self) -> tp.Tuple[pd.DataFrame, pd.Series]:
        state_dict, _ = self.trainer.train()
        model, dataset = self.trainer.init(state_dict=state_dict)
        features = self.trainer.features

        # Call the core function
        importances, spearman = compute_feature_importance(
            model=model,
            dataset=dataset,
            features=features,
            n_perm=self.n_perm,
            alpha=self.alpha,
            warn_ci=self.warn_ci,
        )

        return importances, spearman
