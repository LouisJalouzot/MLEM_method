import typing as tp
from time import time

import numpy as np
import pandas as pd
import torch
from captum.attr import FeaturePermutation
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from torch import nn
from tqdm.auto import tqdm

from src.dataset import Dataset
from src.estimate_correlations import EstimateCorrelations
from src.pairwise_dataloader import PairwiseDataloader
from src.trainer import Trainer
from src.utils import BaseModelSharing, compute_stats, get_device


def compute_feature_importance(
    model: nn.Module,
    dataloader: PairwiseDataloader,
    clusters: pd.DataFrame,
    n_perm: int,
    monitor: tp.Literal["std", "ci_width"] = "std",
    thresh: float = 0.01,
    alpha: float = 0.01,
) -> tp.Tuple[pd.DataFrame, pd.Series]:
    logger.info(
        f"Computing permutation feature importance with {n_perm} permutations "
        f"for {clusters.Cluster.max() + 1} clusters of feature pairs."
    )

    features = clusters.Feature
    clusters = torch.from_numpy(clusters.Cluster.values)

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
        clusters = clusters.to(X_batch.device)

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
        batch_importances = feature_perm.attribute(
            X_batch_flat, feature_mask=clusters[None]
        ).cpu()
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
    importances = pd.DataFrame(importances, columns=features)
    importances = compute_stats(importances, alpha=alpha)
    importances = importances.reset_index(names="Feature")
    importances["Cluster"] = clusters.cpu()
    cols = [col for col in importances.columns if col not in ["Cluster", "Feature"]]
    aggregations = {
        "Feature": [
            ("Feature", lambda x: min(x, key=len)),
            ("AllFeatures", list),
        ]
    } | {col: "first" for col in cols}
    importances = importances.groupby("Cluster").agg(aggregations)
    new_columns = [
        col_tuple[1] if col_tuple[0] == "Feature" else col_tuple[0]
        for col_tuple in importances.columns
    ]
    importances.columns = new_columns
    importances = importances.reset_index()

    importances = importances.sort_values("mean", ascending=False)

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

    return importances, spearman_stats.to_frame().T


class FeatureImportance(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(
        default_factory=lambda: EstimateCorrelations()
    )
    trainer: Trainer = Field(default_factory=lambda: Trainer())

    n_perm: int = 10
    monitor: tp.Literal["std", "ci_width"] = "std"
    thresh: float = 0.01
    alpha: float = 0.01

    device: str | None = None
    infra: TaskInfra = TaskInfra(version="1", folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["trainer", "estimate_correlations"],
        "estimate_correlations": ["trainer"],
    }
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    def model_post_init(self, __context: tp.Any) -> None:
        if self.device is None:
            self.device = get_device()

    @infra.apply
    def compute(self) -> tp.Tuple[pd.DataFrame, pd.Series]:
        state_dict, _ = self.trainer.train()
        model, dataloader = self.trainer.init(state_dict=state_dict, device=self.device)
        model.eval()

        # Estimate correlations with forced product
        ec_params = self.estimate_correlations.model_dump()
        ec_params["product"] = True
        clusters = EstimateCorrelations(**ec_params).cluster_features()

        importances, spearman = compute_feature_importance(
            model=model,
            dataloader=dataloader,
            clusters=clusters,
            n_perm=self.n_perm,
            monitor=self.monitor,
            thresh=self.thresh,
            alpha=self.alpha,
        )

        return importances, spearman
