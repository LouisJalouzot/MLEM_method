import torch
from scipy import stats
from src.pairwise_dataset import PairwiseDataset
from loguru import logger
from pydantic import BaseModel
from exca import TaskInfra
import typing as tp


def batch_corrcoef(
    x: torch.Tensor, ddof: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes batched Pearson correlation coefficient matrix manually.

    Args:
        x: Input tensor of shape (B, N, D). B=batch, N=observations, D=variables/features.
        ddof: Delta Degrees of Freedom (1 for sample, 0 for population).
        eps: Small value for numerical stability.

    Returns:
        Batched correlation matrix of shape (B, D, D).
    """
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


class CorrelationEstimatorConfig(BaseModel):
    bootstrap: int = 5
    init_sample_size: int = 4096
    factor: float = 1.2
    max_sample_size: float = 1e6
    confidence: float = 0.99
    max_margin: float = 5e-2
    product: bool = False
    infra: TaskInfra = TaskInfra(version="1", folder=".cache")

    @infra.apply
    def estimate_corrs(self, X: torch.Tensor) -> tp.Tuple[torch.Tensor, int]:
        """
        Estimates the correlation matrix using a bootstrapping approach.

        The function iteratively increases the sample size until the confidence intervals
        of the correlation coefficients are within a specified margin.

        Args:
            X: Feature tensor of shape (N, D), N=samples, D=variables.

        Returns:
            Tuple containing the estimated correlation matrix of shape (D, D)
            and the sample size used.
        """
        dataset = PairwiseDataset(X)
        i, j = torch.triu_indices(X.shape[1], X.shape[1])
        sample_size = self.init_sample_size

        while sample_size < self.max_sample_size:
            logger.debug(
                f"Estimating correlation with sample size: {sample_size}, bootstrap: {self.bootstrap}"
            )

            # Get samples
            X_batch = dataset.sample(
                self.bootstrap * sample_size, only_valid=False
            )
            if self.product:
                X_batch = (X_batch[:, None] * X_batch[:, :, None])[:, i, j]

            # Reshape for batch processing
            X_batch = X_batch.reshape(self.bootstrap, sample_size, -1)

            # Compute correlations and confidence intervals
            corrs = batch_corrcoef(X_batch)
            cis = compute_ci(
                corrs.reshape(self.bootstrap, -1), confidence=self.confidence
            )
            margins = cis[:, 1] - cis[:, 0]

            logger.debug(
                f"Max margin: {margins.max():.2g} (needs to be < {self.max_margin:.2g})"
            )

            # Check if margin is acceptable
            if margins.max() < self.max_margin:
                logger.info(
                    f"Sample size {sample_size} is sufficient to estimate correlations "
                    f"with the required confidence and precision."
                )
                return corrs.mean(dim=0), sample_size

            # Increase sample size for next iteration
            sample_size = int(sample_size * self.factor) + 1

        raise ValueError(
            f"Could not estimate correlations with the required confidence and precision "
            f"under {self.max_sample_size} samples."
        )
