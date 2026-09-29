import warnings

import numpy as np
import pytest
import torch
from scipy.stats import ConstantInputWarning, spearmanr

from mlem_method.utils import spearman


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spearman_matches_scipy(dtype):
    """Average ranks for ties; NaN for constant or NaN inputs (standard behavior)."""
    rng = np.random.default_rng(1)
    cases = [
        (rng.normal(size=64), rng.normal(size=64)),
        (rng.integers(0, 5, 64).astype(float), rng.integers(0, 5, 64).astype(float)),
        (np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 2.0])),
        (np.ones(8), np.arange(8.0)),
        (np.ones(8), np.ones(8)),
        (rng.normal(size=(3, 64)), rng.normal(size=(3, 64))),
        (np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 2.0])),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for a, b in cases:
            x, y = torch.tensor(a, dtype=dtype), torch.tensor(b, dtype=dtype)
            aa, bb = np.broadcast_arrays(a, b)
            expected = np.array(
                [
                    spearmanr(row_a, row_b).statistic
                    for row_a, row_b in zip(aa.reshape(-1, aa.shape[-1]), bb.reshape(-1, bb.shape[-1]))
                ]
            ).reshape(aa.shape[:-1])
            actual = spearman(x, y)
            assert isinstance(actual, torch.Tensor) and actual.device == x.device
            assert actual.dtype == dtype
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5, equal_nan=True)
