from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field

from mlem.dataset import Dataset
from mlem.pairwise_dataloader import PairwiseDataloader, PairwiseDataloaderBuilder
from mlem.utils import BaseModel, get_device, seed_from_basemodel


def batch_corrcoef(x: torch.Tensor, ddof: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes batched Pearson correlation coefficient matrix manually.

    Args:
        x: Input tensor of shape (B, N, D). B=batch, N=observations, D=variables/features.
        ddof: Delta Degrees of Freedom (1 for sample, 0 for population).
        eps: Small value for numerical stability.

    Returns:
        Batched correlation matrix of shape (B, D, D).
    """
    import torch

    B, N, D = x.shape
    if N <= ddof:
        raise ValueError(
            f"Number of observations N={N} must be greater than ddof={ddof}"
        )

    # Center data: (B, N, D)
    mean = torch.mean(x, dim=1, keepdim=True)
    x_centered = x - mean

    # Compute covariance numerator: (B, D, N) @ (B, N, D) -> (B, D, D)
    covariance_numerator = x_centered.transpose(1, 2) @ x_centered

    # Compute standard deviations: (B, 1, D)
    std_dev = torch.std(x, dim=1, unbiased=(ddof == 1), keepdim=True)

    # Compute denominator: (B, D, 1) @ (B, 1, D) -> (B, D, D)
    denominator = std_dev.transpose(1, 2) @ std_dev

    # Compute correlation matrix with stability term
    corr_matrix = covariance_numerator / (denominator * (N - ddof) + eps)

    # Clamp results to [-1, 1]
    return torch.clamp(corr_matrix, -1.0, 1.0)


def compute_ci(data: torch.Tensor, confidence: float = 0.99) -> torch.Tensor:
    """
    Computes the confidence interval for each variable using Z-score.

    Args:
        data: Input tensor of shape (N, T), N=samples, T=variables. Assumes N > 1.
        confidence: The desired confidence level (e.g., 0.99 for 99%).

    Returns:
        Tensor of shape (2, T) with lower and upper bounds for each variable's mean.
    """
    import torch
    from scipy import stats

    n = data.shape[0]
    # Handle case with 1 or 0 samples, returning NaNs
    if n <= 1 or not (0 < confidence < 1):
        return torch.full(
            (2, data.shape[1]),
            float("nan"),
            dtype=data.dtype,
            device=data.device,
        )

    mean = torch.mean(data, dim=0)
    std_dev = torch.std(data, dim=0, unbiased=True)
    se = std_dev / n**0.5

    # Calculate Z-score for the given confidence level
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha / 2.0)

    # Margin of error
    margin = z_score * se

    # Confidence interval [lower, upper]
    lower_bound = mean - margin
    upper_bound = mean + margin

    return torch.stack([lower_bound, upper_bound], dim=1)


def estimate_correlations(
    dataloader: PairwiseDataloader,
    n_trials: int = 64,
    init_sample_size: int = 4096,
    factor: float = 1.2,
    max_sample_size: float = 2**20,
    product: bool = False,
    monitor: tp.Literal["std", "ci_width"] = "std",
    thresh: float = 0.01,
    ci_confidence: float = 0.99,
) -> tp.Tuple[torch.Tensor, int]:
    import torch

    _, n_features = dataloader.get_X_shape()
    triu_indices = torch.triu_indices(n_features, n_features)
    sample_size = init_sample_size

    while sample_size < max_sample_size:
        logger.debug(
            f"Estimating correlation with sample size: {sample_size}, n_trials: {n_trials}"
        )

        # Get samples
        # (n_trials, n_samples, n_features)
        X_batch = dataloader.sample(n_pairs=sample_size, n_trials=n_trials)
        if product:
            # (n_trials, n_samples, n_features, n_features)
            X_batch = X_batch[:, :, None] * X_batch[:, :, :, None]
            # (n_trials, n_samples, n_feature_pairs)
            X_batch = X_batch[:, :, *triu_indices]

        # Compute correlations and confidence intervals
        # (n_trials, n_feature_pairs)
        corrs = batch_corrcoef(X_batch)
        if monitor == "std":
            # (n_feature_pairs,)
            stds = corrs.std(dim=0)
            variability = stds.max()
            logger.debug(f"Max std: {variability:<4.2g} (needs to be < {thresh:.2g})")
        elif monitor == "ci_width":
            cis = compute_ci(corrs.reshape(n_trials, -1), confidence=ci_confidence)
            variability = (cis[:, 1] - cis[:, 0]).max()
            logger.debug(
                f"Max CI width: {variability:<4.2g} (needs to be < {thresh:.2g})"
            )

        # Check if variability is acceptable
        if variability < thresh:
            logger.info(
                f"Sample size {sample_size} is sufficient to estimate correlations "
                f"with the required variability."
            )
            return corrs.mean(dim=0).cpu(), sample_size

        # Increase sample size for next iteration
        sample_size = int(sample_size * factor) + 1

    raise ValueError(
        f"Could not estimate correlations with the required variability "
        f"under {max_sample_size} samples."
    )


class EstimateCorrelations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    dataloader_builder: PairwiseDataloaderBuilder = Field(
        default_factory=lambda: PairwiseDataloaderBuilder()
    )
    n_trials: int = 64
    init_sample_size: int = 4096
    factor: float = 1.2
    max_sample_size: float = 2**20
    product: bool = True
    monitor: tp.Literal["std", "ci_width"] = "std"
    thresh: float = 0.01
    ci_confidence: float = 0.99

    clustering_linkage: tp.Literal["single", "complete", "average", "ward"] = "single"
    clustering_threshold: float = 0

    device: tp.Optional[str] = None
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    @infra.apply
    def estimate_correlations(self) -> tp.Tuple[torch.Tensor, int]:
        import pandas as pd

        X = self.dataset.encode().to(self.device or get_device())
        dataloader = self.dataloader_builder.build_for_estimation(
            X, seed=seed_from_basemodel(self)
        )

        correlations, n_pairs = estimate_correlations(
            dataloader=dataloader,
            n_trials=self.n_trials,
            init_sample_size=self.init_sample_size,
            factor=self.factor,
            max_sample_size=self.max_sample_size,
            product=self.product,
            monitor=self.monitor,
            thresh=self.thresh,
            ci_confidence=self.ci_confidence,
        )
        if self.product:
            labels = self.dataset.pfeatures
        else:
            labels = self.dataset.features
        correlations = pd.DataFrame(
            correlations,
            columns=labels,
            index=labels,
        )

        return correlations, n_pairs

    def cluster_features(self) -> pd.DataFrame:
        import pandas as pd
        from sklearn.cluster import AgglomerativeClustering

        correlations, _ = self.estimate_correlations()
        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage=self.clustering_linkage,
            distance_threshold=self.clustering_threshold,
            n_clusters=None,
        )
        clusters = clustering.fit_predict(1 - abs(correlations))
        logger.info(
            f"{len(correlations)} feature{' pairs' if self.product else 's'} clustered into {clusters.max() + 1} clusters."
        )

        return pd.DataFrame({"Cluster": clusters, "Feature": correlations.columns})
