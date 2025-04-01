import torch
from scipy import stats
from src.pairwise_dataset import PairwiseDataset
from loguru import logger


def batch_corrcoef(
    x: torch.Tensor, ddof: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes batched Pearson correlation coefficient matrix manually.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, D).
                          B=batch, N=observations, D=variables/features.
        ddof (int): Delta Degrees of Freedom (1 for sample, 0 for population).
        eps (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Batched correlation matrix of shape (B, D, D).
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
    # Note: Using covariance_numerator / (N - ddof) for cov_matrix first, then dividing,
    # is equivalent to the line below and avoids redundant division/multiplication.
    corr_matrix = covariance_numerator / (denominator * (N - ddof) + eps)

    # Clamp results to [-1, 1]
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    return corr_matrix


def compute_ci(data: torch.Tensor, confidence: float = 0.99) -> torch.Tensor:
    """
    Computes the confidence interval for each variable (column) using Z-score.

    Args:
        data (torch.Tensor): Input tensor of shape (N, T), N=samples, T=variables.
                               Assumes N > 1.
        confidence (float): The desired confidence level (e.g., 0.99 for 99%).

    Returns:
        torch.Tensor: Tensor of shape (2, T) where row 0 is the lower bound
                      and row 1 is the upper bound for each variable's mean.
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
    # Sample std dev (N-1 denominator)
    std_dev = torch.std(data, dim=0, unbiased=True)

    # Standard error of the mean
    se = std_dev / n**0.5

    # Calculate Z-score for the given confidence level
    alpha = 1 - confidence
    # Use scipy.stats.norm.ppf for inverse CDF (quantile function)
    z_score = stats.norm.ppf(1 - alpha / 2.0)

    # Margin of error
    margin = z_score * se

    # Confidence interval [lower, upper]
    lower_bound = mean - margin
    upper_bound = mean + margin

    return torch.stack([lower_bound, upper_bound], dim=1)


def estimate_corrs(
    X: torch.Tensor,
    product: bool = False,
    bootstrap: int = 5,
    init_sample_size: int = 4096,
    factor: float = 1.2,
    max_sample_size: float = 1e6,
    confidence: float = 0.99,
    max_margin: float = 5e-2,
) -> tuple[torch.Tensor, int]:
    """
    Estimates the correlation matrix of a given tensor X using a bootstrapping approach.

    The function iteratively increases the sample size until the confidence intervals
    of the correlation coefficients are within a specified margin.

    Args:
        X (torch.Tensor): Feature array encoded with src.encode_df
                          Tensor of shape (N, D), where N is the number of
                          samples and D is the number of variables.
        bootstrap (int): Number of bootstrap samples to use for estimating the
                         correlation coefficients.
        init_sample_size (int): Initial sample size to start with.
        max_sample_size (int): Maximum sample size to use. The function will stop
                               and raise an error if the required confidence and
                               precision are not achieved with this sample size.
        confidence (float): Confidence level for the confidence intervals.
        max_margin (float): Maximum allowed margin of error for the confidence intervals.

    Returns:
        tuple[torch.Tensor, int]: A tuple containing:
            - The estimated correlation matrix of shape (D, D).
            - The sample size used to estimate the correlation matrix.

    Raises:
        ValueError: If the required confidence and precision are not achieved within
                    the maximum allowed sample size.
    """
    dataset = PairwiseDataset(X)
    i, j = torch.triu_indices(X.shape[1], X.shape[1])
    sample_size = init_sample_size
    while sample_size < max_sample_size:
        logger.debug(
            f"Estimating correlation with sample size: {sample_size} and bootstrap: {bootstrap}"
        )
        X_batch = dataset.sample(bootstrap * sample_size, only_valid=False)
        if product:
            X_batch = (X_batch[:, None] * X_batch[:, :, None])[:, i, j]
        X_batch = X_batch.reshape(bootstrap, sample_size, -1)
        corrs = batch_corrcoef(X_batch)
        cis = compute_ci(corrs.reshape(bootstrap, -1), confidence=confidence)
        margins = cis[:, 1] - cis[:, 0]
        logger.debug(
            f"Max margin: {margins.max():.2g} (needs to be < {max_margin:.2g})"
        )
        if margins.max() < max_margin:
            logger.info(
                f"Sample size {sample_size} is sufficient to estimate correlations with the required confidence and precision."
            )
            return corrs.mean(dim=0), sample_size
        sample_size = int(sample_size * factor) + 1

    raise ValueError(
        f"Could not estimate correlations with the required confidence and precision under {max_sample_size} samples."
    )
