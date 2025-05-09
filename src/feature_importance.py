import typing as tp
from time import time

import numpy as np
import pandas as pd
import torch
from captum.attr import FeaturePermutation
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from statsmodels.stats.descriptivestats import describe
from torch import nn
from tqdm.auto import tqdm

from src.dataset import Dataset
from src.estimate_correlations import EstimateCorrelations
from src.pairwise_dataloader import PairwiseDataloader
from src.trainer import Trainer
from src.utils import BaseModelSharing


def compute_stats(data, alpha=0.01):
    """Compute descriptive statistics with confidence intervals"""
    return describe(
        data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha
    ).T


def compute_feature_importance(
    model: nn.Module,
    dataloader: PairwiseDataloader,
    features: list[str],
    n_perm: int,
    monitor: tp.Literal["std", "ci_width"] = "std",
    thresh: float = 0.01,
    alpha: float = 0.01,
) -> tp.Tuple[pd.DataFrame, pd.Series]:
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

    for i in tqdm(
        range(1, n_perm + 1),
        desc="Computing feature importance",
        disable=True,
    ):
        t = time()
        X_batch, Y_batch = dataloader[i]

        # Create flattened feature interactions
        X_batch_flat = X_batch[:, None] * X_batch[:, :, None]
        X_batch_flat = X_batch_flat[:, *model.triu_indices]

        # Define the function to measure attributions
        def f(x):
            with torch.no_grad():
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
            f"Duration: {time() - t:<8.3f}s - "
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
    if monitor == "ci_width":
        variability = spearman_stats["upper_ci"] - spearman_stats["lower_ci"]
        message = f"the width of the {(1 - alpha)*100:.3g}% confidence interval of the Spearman correlation is {variability:.3g} "
    elif monitor == "std":
        variability = spearman_stats["std"]
        message = (
            f"the standard deviation of the Spearman correlation is {variability:.3g} "
        )
    if variability > thresh:
        logger.warning(
            f"Significant variability between batches: "
            + message
            + f"which is larger than the threshold {thresh:.3g}."
        )

    return importances_stats, spearman_stats


class FeatureImportance(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(
        default_factory=lambda: EstimateCorrelations()
    )
    trainer: Trainer = Field(default_factory=lambda: Trainer())

    n_perm: int = 30
    monitor: tp.Literal["std", "ci_width"] = "std"
    thresh: float = 0.01
    alpha: float = 0.01

    infra: TaskInfra = TaskInfra(version="1", folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["trainer"],
        "estimate_correlations": ["trainer"],
        "infra": ["dataset", "estimate_correlations", "trainer"],
    }

    @infra.apply
    def compute(self) -> tp.Tuple[pd.DataFrame, pd.Series]:
        state_dict, _ = self.trainer.train()
        model, dataloader = self.trainer.init(state_dict=state_dict)
        features = self.dataset.features

        importances, spearman = compute_feature_importance(
            model=model,
            dataloader=dataloader,
            features=features,
            n_perm=self.n_perm,
            monitor=self.monitor,
            thresh=self.thresh,
            alpha=self.alpha,
        )

        return importances, spearman
