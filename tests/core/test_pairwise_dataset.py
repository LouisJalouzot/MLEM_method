import pytest
import torch

from src.core.pairwise_dataset import PairwiseDataset


@pytest.fixture
def pairwise_data():
    n = 10
    n_features = 5
    X = torch.randn(n, n_features)
    # Add NaNs to test nan_to_num
    X[0, 0] = torch.nan
    Y = torch.randn(n, 768)  # Example embedding dimension
    return X, Y


def test_pairwise_dataset_init(pairwise_data):
    X, Y = pairwise_data
    dataset = PairwiseDataset(X=X, Y=Y, n_pairs=100)
    assert dataset.n == len(X)
    assert dataset.n_features == X.shape[1]
    assert dataset.n_pairs == 100


def test_pairwise_dataset_init_no_x(pairwise_data):
    _, Y = pairwise_data
    dataset = PairwiseDataset(Y=Y, n_pairs=100)
    assert dataset.n == len(Y)
    assert dataset.X is None


def test_pairwise_dataset_init_no_y(pairwise_data):
    X, _ = pairwise_data
    dataset = PairwiseDataset(X=X, n_pairs=100)
    assert dataset.n == len(X)
    assert dataset.n_features == X.shape[1]
    assert dataset.Y is None


def test_pairwise_dataset_init_no_data():
    with pytest.raises(ValueError):
        PairwiseDataset(n_pairs=100)


def test_pairwise_dataset_sample_shapes(pairwise_data):
    X, Y = pairwise_data
    n_pairs_sample = 50
    dataset = PairwiseDataset(X=X, Y=Y, n_pairs=n_pairs_sample)
    X_dist, Y_dist = dataset.sample(n_pairs=n_pairs_sample)

    assert (
        X_dist.shape[0] <= n_pairs_sample
    )  # Can be less due to removing i==j pairs
    assert X_dist.shape[1] == X.shape[1]
    assert (
        Y_dist.shape[0] == X_dist.shape[0]
    )  # Should have the same number of pairs


def test_pairwise_dataset_sample_only_x(pairwise_data):
    X, _ = pairwise_data
    n_pairs_sample = 50
    dataset = PairwiseDataset(X=X, n_pairs=n_pairs_sample)
    X_dist = dataset.sample(n_pairs=n_pairs_sample)
    assert X_dist.shape[0] <= n_pairs_sample
    assert X_dist.shape[1] == X.shape[1]


def test_pairwise_dataset_sample_only_y(pairwise_data):
    _, Y = pairwise_data
    n_pairs_sample = 50
    dataset = PairwiseDataset(Y=Y, n_pairs=n_pairs_sample)
    Y_dist = dataset.sample(n_pairs=n_pairs_sample)
    assert Y_dist.shape[0] <= n_pairs_sample


def test_pairwise_dataset_getitem(pairwise_data):
    X, Y = pairwise_data
    n_pairs_base = 100
    gamma = 0.9
    dataset = PairwiseDataset(X=X, Y=Y, n_pairs=n_pairs_base, gamma=gamma)

    # Test first item
    X_dist_0, Y_dist_0 = dataset[0]
    expected_pairs_0 = int(n_pairs_base * (gamma**0))
    assert X_dist_0.shape[0] <= expected_pairs_0
    assert Y_dist_0.shape[0] == X_dist_0.shape[0]

    # Test second item
    X_dist_1, Y_dist_1 = dataset[1]
    expected_pairs_1 = int(n_pairs_base * (gamma**1))
    assert X_dist_1.shape[0] <= expected_pairs_1
    assert Y_dist_1.shape[0] == X_dist_1.shape[0]


def test_pairwise_dataset_getitem_gamma(pairwise_data):
    X, Y = pairwise_data
    n_pairs_base = 100
    gamma = 0.9
    dataset = PairwiseDataset(X=X, Y=Y, n_pairs=n_pairs_base, gamma=gamma)

    # Test first item (gamma^0 = 1)
    X_dist_0, Y_dist_0 = dataset[0]
    expected_pairs_0 = int(n_pairs_base * (gamma**0))
    # Check if the number of pairs is close to expected (allowing for removal of i==j)
    assert X_dist_0.shape[0] <= expected_pairs_0
    assert Y_dist_0.shape[0] == X_dist_0.shape[0]

    # Test second item (gamma^1 = 0.9)
    X_dist_1, Y_dist_1 = dataset[1]
    expected_pairs_1 = int(n_pairs_base * (gamma**1))
    assert X_dist_1.shape[0] <= expected_pairs_1
    assert Y_dist_1.shape[0] == X_dist_1.shape[0]

    # Test third item (gamma^2 = 0.81)
    X_dist_2, Y_dist_2 = dataset[2]
    expected_pairs_2 = int(n_pairs_base * (gamma**2))
    assert X_dist_2.shape[0] <= expected_pairs_2
    assert Y_dist_2.shape[0] == X_dist_2.shape[0]


def test_pairwise_dataset_nan_to_num(pairwise_data):
    X, Y = pairwise_data  # X now contains a NaN
    nan_replacement_value = -99.0
    dataset = PairwiseDataset(
        X=X, Y=Y, n_pairs=50, nan_to_num=nan_replacement_value
    )

    X_dist, Y_dist = dataset.sample(n_pairs=50)

    # Check if NaNs in the original X resulted in the replacement value in X_dist
    # X_dist = abs(X1 - X2).nan_to_num(val).clip(0, 1)
    # If X1 or X2 had NaN, the difference is NaN, then replaced by nan_replacement_value
    # Then abs() is applied, then clipped.
    # So, we expect the replacement value (or its absolute if negative) if NaN was involved,
    # unless the clip(0,1) changed it.
    # A simpler check is just to ensure no NaNs are present in the output X_dist.
    assert not torch.isnan(X_dist).any()

    # Check if the specific nan_replacement_value appears (if it's within [0, 1] after abs/clip)
    # This is hard to guarantee. Let's check the source again.
    # X_dist = (X_1 - X_2).nan_to_num(self.nan_to_num).abs().clip(0, 1)
    # If nan_to_num is 0 (default), NaNs become 0. If it's -99, NaNs become abs(-99)=99, then clipped to 1.
    dataset_nan_zero = PairwiseDataset(X=X, Y=Y, n_pairs=50, nan_to_num=0)
    X_dist_zero, _ = dataset_nan_zero.sample(n_pairs=50)
    assert not torch.isnan(X_dist_zero).any()
    # Check if 0 appears more often than expected if NaNs weren't replaced by 0

    dataset_nan_one = PairwiseDataset(
        X=X, Y=Y, n_pairs=50, nan_to_num=99
    )  # Becomes 1 after clip
    X_dist_one, _ = dataset_nan_one.sample(n_pairs=50)
    assert not torch.isnan(X_dist_one).any()
    # Check if 1 appears more often than expected if NaNs weren't replaced by 1


def test_pairwise_dataset_distance_cosine(pairwise_data):
    X, Y = pairwise_data
    dataset = PairwiseDataset(
        Y=Y, distance="cosine", min_max_scale=False
    )  # Test without scaling first
    Y_dist = dataset.sample(n_pairs=10)
    # Cosine distance is 1 - sim. Sim is [-1, 1]. So distance is [0, 2].
    assert torch.all(Y_dist >= 0) and torch.all(Y_dist <= 2)

    dataset_scaled = PairwiseDataset(Y=Y, distance="cosine", min_max_scale=True)
    Y_dist_scaled = dataset_scaled.sample(
        n_pairs=50
    )  # Need enough samples for scaling
    # With scaling, should be [0, 1]
    assert torch.all(Y_dist_scaled >= 0) and torch.all(Y_dist_scaled <= 1)


def test_pairwise_dataset_distance_l1(pairwise_data):
    X, Y = pairwise_data
    dataset = PairwiseDataset(Y=Y, distance=1)  # L1 distance
    Y_dist = dataset.sample(n_pairs=10)
    assert torch.all(Y_dist >= 0)


def test_pairwise_dataset_min_max_scaling(pairwise_data):
    X, Y = pairwise_data
    dataset_scaled = PairwiseDataset(Y=Y, min_max_scale=True)
    Y_dist_scaled = dataset_scaled.sample(n_pairs=50)
    # With enough samples, min should approach 0 and max approach 1
    assert torch.all(Y_dist_scaled >= 0) and torch.all(Y_dist_scaled <= 1)

    dataset_unscaled = PairwiseDataset(Y=Y, min_max_scale=False)
    Y_dist_unscaled = dataset_unscaled.sample(n_pairs=50)
    # Unscaled distances should just be >= 0
    assert torch.all(Y_dist_unscaled >= 0)
    # Check if max is likely > 1 (unless all vectors are identical)
    if not torch.allclose(Y[0], Y[1]):
        assert torch.max(Y_dist_unscaled) > 1.0
