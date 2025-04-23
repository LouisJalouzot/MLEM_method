import numpy as np
import pandas as pd
import pytest
import torch
from statsmodels.stats.descriptivestats import (  # Import describe for comparison if needed
    describe,
)

# Assuming necessary imports from src.core
from src.core.feature_importance import (
    compute_feature_importance_core,
    compute_stats,
)
from src.core.pairwise_dataset import (
    PairwiseDataset,
)  # Mock or simple version needed
from src.core.spd_matrix_learner import (  # Mock or simple version needed
    SPDMatrixLearner,
)

# Remove the local definition of compute_stats as we import the real one now


def test_compute_stats():
    # Basic test for compute_stats using the imported function
    data = pd.DataFrame(
        {"col1": np.random.rand(10), "col2": np.random.rand(10)}
    )
    stats = compute_stats(data, alpha=0.05)  # Use the imported function
    assert isinstance(stats, pd.DataFrame)
    assert "mean" in stats.columns
    assert "std" in stats.columns
    assert "lower_ci" in stats.columns
    assert "upper_ci" in stats.columns
    assert stats.index.tolist() == ["col1", "col2"]


# Mock classes/functions needed for compute_feature_importance_core
class MockSPDMatrixLearner:
    def __init__(self, n_features):
        self.n_features = n_features
        # Use torch.triu_indices consistent with the source code
        self.triu_indices = torch.triu_indices(n_features, n_features)

    def flat_forward(self, x):
        # Simple mock forward pass - returns a scalar score per sample
        # The actual function expects a scalar output for the spearman calculation
        return torch.rand(x.shape[0])

    def spearman(self, pred, target):
        # Simple mock spearman calculation - returns a single scalar value
        return torch.tensor(0.5 + torch.rand(1).item() * 0.1)  # Add some noise

    def __call__(self, x):
        # Mock __call__ needed for baseline performance calculation inside the loop
        # It should take the original features (X_batch) and return pairwise predictions
        n_samples = x.shape[0]
        n_pairs = n_samples * (n_samples - 1) // 2
        return torch.rand(n_pairs)


class MockPairwiseDataset:
    def __init__(self, n_samples, n_features, n_perms):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_perms = n_perms
        # Generate some fixed data for reproducibility if needed
        self.X_data = [
            torch.rand(n_samples, n_features) for _ in range(n_perms)
        ]
        self.Y_data = [
            torch.rand(n_samples * (n_samples - 1) // 2) for _ in range(n_perms)
        ]

    def __getitem__(self, index):
        # Return mock data for a batch
        if index >= self.n_perms:
            raise IndexError("Index out of bounds")
        # Return pre-generated data
        X = self.X_data[index]
        Y = self.Y_data[index]
        return X, Y

    def __len__(self):
        return self.n_perms


# Unskip the test and implement basic assertions
# @pytest.mark.skip(reason="Requires complex mocking or setup") # Unskip
def test_compute_feature_importance_core():
    # Basic test structure for compute_feature_importance_core
    n_features = 5
    n_samples = 10
    n_perm = 3  # Keep small for testing
    features = [f"feat_{i}" for i in range(n_features)]
    alpha = 0.05
    warn_ci = 0.1

    # Mock model and dataset
    mock_model = MockSPDMatrixLearner(n_features)
    mock_dataset = MockPairwiseDataset(n_samples, n_features, n_perm)

    # Call the function (assuming it's imported)
    importances_stats, spearman_stats = compute_feature_importance_core(
        model=mock_model,
        dataset=mock_dataset,
        features=features,
        n_perm=n_perm,
        alpha=alpha,
        warn_ci=warn_ci,
    )

    # Add assertions based on expected output structure
    assert isinstance(importances_stats, pd.DataFrame)
    assert isinstance(spearman_stats, pd.Series)
    assert "Feature" in importances_stats.columns
    assert "mean" in importances_stats.columns
    assert "std" in importances_stats.columns
    assert "ci" in importances_stats.columns  # Check for 'ci' tuple
    # Expected number of feature pairs (upper triangle including diagonal)
    expected_num_pfeatures = n_features * (n_features + 1) // 2
    assert len(importances_stats) == expected_num_pfeatures

    assert "mean" in spearman_stats.index
    assert "std" in spearman_stats.index
    assert "ci" in spearman_stats.index  # Check for 'ci' tuple
