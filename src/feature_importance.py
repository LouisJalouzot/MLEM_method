from __future__ import annotations

import typing as tp
from time import time

import numpy as np
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from src.dataset import Dataset
from src.estimate_correlations import EstimateCorrelations
from src.pairwise_dataloader import PairwiseDataloader
from src.trainer import Trainer
from src.utils import BaseModelSharing, compute_stats

if tp.TYPE_CHECKING:
    import pandas as pd
    from torch import nn


def compute_feature_importance(
    model: nn.Module,
    dataloader: PairwiseDataloader,
    clusters: pd.DataFrame,
    n_perm: int = 5,
    monitor: tp.Literal["std", "ci_width"] = "std",
    thresh: float = 0.01,
    alpha: float = 0.01,
) -> tp.Tuple[pd.DataFrame, pd.Series]:
    import pandas as pd
    import torch

    device = dataloader.X.device
    cluster_ids = torch.from_numpy(clusters.Cluster.values).to(device)
    n_clusters = cluster_ids.max().item() + 1

    # Storage for results
    importances = {}
    score = []
    start = time()

    pbar = tqdm(
        total=n_clusters * n_perm, desc="Computing Permutation Feature Importance"
    )
    with pbar:
        for c in range(n_clusters):
            mask = cluster_ids == c
            importances[c] = []
            for _ in range(n_perm):
                X_batch, Y_batch = dataloader.sample()
                # Create flattened feature interactions
                X_batch_flat = X_batch[:, None] * X_batch[:, :, None]
                X_batch_flat = X_batch_flat[:, *model.triu_indices]

                # Compute baseline performance
                baseline_score = model.score(X_batch_flat, Y_batch)
                score.append(baseline_score)

                # Permute features in the cluster
                perm = torch.randperm(X_batch_flat.shape[0])
                X_batch_flat[:, mask] = X_batch_flat[perm][:, mask]

                # Compute score with permuted data
                permuted_score = model.score(X_batch_flat, Y_batch)
                importance = baseline_score - permuted_score
                if not model.maximize:
                    importance *= -1
                importances[c].append(importance)

                pbar.update(1)

    # Compute importances statistics
    importances = pd.DataFrame(importances)
    importances = compute_stats(importances, alpha=alpha)
    importances = importances.reset_index(names="Cluster")
    importances["Feature"] = clusters.groupby("Cluster").Feature.apply(
        lambda x: min(x, key=len)
    )
    importances["AllFeatures"] = clusters.groupby("Cluster").Feature.apply(list)
    importances = importances.reset_index().sort_values("mean", ascending=False)

    # Compute score statistics
    score_stats = compute_stats(score, alpha=alpha).iloc[0]

    # Log results
    logger.info(
        f"Feature importance computed in {time() - start:.3g}s. "
        f"Mean score = {score_stats['mean']:.3g} ± {score_stats['std']:.3g}"
    )

    # Warn if there's significant variability on the score correlation
    # across batches
    if monitor == "ci_width":
        variability = score_stats["upper_ci"] - score_stats["lower_ci"]
        message = f"the width of the {(1 - alpha)*100:.3g}% confidence interval of the score correlation is {variability:.3g} "
    elif monitor == "std":
        variability = score_stats["std"]
        message = f"the standard deviation of the score correlation is {variability:.3g} "
    if variability > thresh:
        logger.warning(
            f"Significant variability between batches: "
            + message
            + f"which is larger than the threshold {thresh:.3g}."
        )

    return importances, score_stats.to_frame().T


def compute_cv_stats_per_split(df, alpha=0.01):
    import pandas as pd

    values = "mean" if "mean" in df.columns else "Weight"
    df[values] = df[values].astype(float)
    if "AllFeatures" in df.columns:
        all_features = df[["Feature", "AllFeatures"]].drop_duplicates("Feature")
    else:
        all_features = None
    if "Feature" in df.columns:
        df_pivot = df.pivot(index=["cv", "split"], columns="Feature", values=values)
        gb = df_pivot.groupby("split")
    else:
        gb = df.groupby("split")["mean"]

    all_stats = []
    for split, group in gb:
        stats = compute_stats(group, alpha)
        stats["split"] = split
        all_stats.append(stats.reset_index())
    all_stats = pd.concat(all_stats, ignore_index=True)

    if all_features is not None:
        all_stats = all_features.merge(all_stats)

    return all_stats.sort_values("mean", ascending=False)


class FeatureImportance(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(
        default_factory=lambda: EstimateCorrelations()
    )
    trainer: Trainer = Field(default_factory=lambda: Trainer())

    n_perm: int = 5
    monitor: tp.Literal["std", "ci_width"] = "std"
    thresh: float = 0.01
    alpha: float = 0.01

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["trainer", "estimate_correlations"],
        "estimate_correlations": ["trainer"],
    }

    @infra.apply
    def compute(self) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        import pandas as pd

        # Estimate correlations with forced product
        ec_params = self.estimate_correlations.model_dump()
        ec_params["product"] = True
        clusters = EstimateCorrelations(**ec_params).cluster_features()

        all_models, all_logs = self.trainer.train()

        logger.info(
            f"Computing permutation feature importance with {self.n_perm} permutations "
            f"for {clusters.Cluster.max() + 1} clusters of feature pairs."
        )

        all_importances = []
        all_score = []
        all_weights = []

        for i, (model, logs, (train_dl, test_dl)) in enumerate(
            zip(all_models, all_logs, self.trainer.get_folds())
        ):
            weights = model.get_flat_forwatted_W(pfeatures=self.dataset.pfeatures)
            weights["cv"] = i
            weights["split"] = "train"
            weights["converged"] = False if logs.empty else logs.converged.iloc[0]
            weights["spd"] = False if logs.empty else logs.spd.iloc[0]
            weights["training_duration"] = (
                0 if logs.empty else logs["Step Duration"].sum()
            )
            weights["n_epochs"] = len(logs)
            if hasattr(self.trainer.representations, "gt_weights"):
                weights = weights.merge(self.trainer.representations.gt_weights)
                weights["L2"] = np.linalg.norm(weights.GTWeight - weights.Weight)
            all_weights.append(weights)
            for dl, split in [(train_dl, "train"), (test_dl, "test")]:
                importances, score = compute_feature_importance(
                    model=model,
                    dataloader=dl,
                    clusters=clusters,
                    n_perm=self.n_perm,
                    monitor=self.monitor,
                    thresh=self.thresh,
                    alpha=self.alpha,
                )
                for e in [importances, score]:
                    e["cv"] = i
                    e["split"] = split
                all_importances.append(importances)
                all_score.append(score)

        all_importances = pd.concat(all_importances)
        all_score = pd.concat(all_score)
        all_weights = pd.concat(all_weights)

        return all_importances, all_score, all_weights

    def compute_and_aggregate(self) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_importances, all_score, all_weights = self.compute()

        if all_importances.cv.nunique() > 1:
            all_importances = compute_cv_stats_per_split(
                all_importances, alpha=self.alpha
            )
            all_score = compute_cv_stats_per_split(all_score, alpha=self.alpha)
            all_weights = compute_cv_stats_per_split(all_weights, alpha=self.alpha)

        return all_importances, all_score, all_weights
