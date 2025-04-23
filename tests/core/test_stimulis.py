import numpy as np
import pandas as pd
import pytest
import torch

from src.core.stimulis import encode_df


def test_encode_df_basic():
    """Test encode_df with a simple DataFrame."""
    data = {
        "col_a": [1, 2, 3],
        "col_b": [10.5, 20.5, 30.5],
        "col_c": ["X", "Y", "X"],
    }
    df = pd.DataFrame(data)

    encoded_tensor = encode_df(df)

    assert isinstance(encoded_tensor, torch.Tensor)
    assert encoded_tensor.shape == (3, 3)
    assert encoded_tensor.dtype == torch.float32

    # Check numerical columns scaling (MinMaxScaler scales to [0, 1])
    assert torch.allclose(encoded_tensor[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(encoded_tensor[:, 1], torch.tensor([0.0, 0.5, 1.0]))

    # Check categorical column encoding (should be 0 and 1)
    # Note: Category codes depend on the order pandas assigns them ('X' -> 0, 'Y' -> 1)
    assert torch.allclose(encoded_tensor[:, 2], torch.tensor([0.0, 1.0, 0.0]))


def test_encode_df_with_nan():
    """Test encode_df with NaN values."""
    data = {
        "col_a": [1, np.nan, 3],
        "col_b": [10.5, 20.5, np.nan],
        "col_c": ["X", "Y", None],
    }
    df = pd.DataFrame(data)

    encoded_tensor = encode_df(df)

    assert isinstance(encoded_tensor, torch.Tensor)
    assert encoded_tensor.shape == (3, 3)
    assert encoded_tensor.dtype == torch.float32

    # Check for NaNs in the output tensor (MinMaxScaler handles NaNs)
    # Categorical NaNs become NaN after encoding (-1 code -> np.nan)
    assert torch.isnan(encoded_tensor[1, 0])  # NaN in numerical col_a
    assert torch.isnan(encoded_tensor[2, 1])  # NaN in numerical col_b
    assert torch.isnan(encoded_tensor[2, 2])  # NaN in categorical col_c

    # Check non-NaN values are scaled/encoded correctly
    # Need to handle NaNs for scaling check
    valid_a = encoded_tensor[[0, 2], 0]
    assert torch.allclose(valid_a, torch.tensor([0.0, 1.0]))  # Scaled [1, 3]

    valid_b = encoded_tensor[[0, 1], 1]
    assert torch.allclose(
        valid_b, torch.tensor([0.0, 1.0])
    )  # Scaled [10.5, 20.5]

    valid_c = encoded_tensor[[0, 1], 2]
    assert torch.allclose(
        valid_c, torch.tensor([0.0, 1.0])
    )  # Encoded ['X', 'Y']


def test_encode_df_single_column():
    """Test encode_df with a single column DataFrame."""
    data = {"col_a": [10, 20, 5]}
    df = pd.DataFrame(data)
    encoded_tensor = encode_df(df)
    assert encoded_tensor.shape == (3, 1)
    assert torch.allclose(
        encoded_tensor[:, 0], torch.tensor([0.33333334, 1.0, 0.0])
    )  # Scaled [10, 20, 5]

    data_cat = {"col_c": ["A", "B", "A"]}
    df_cat = pd.DataFrame(data_cat)
    encoded_tensor_cat = encode_df(df_cat)
    assert encoded_tensor_cat.shape == (3, 1)
    assert torch.allclose(
        encoded_tensor_cat[:, 0], torch.tensor([0.0, 1.0, 0.0])
    )  # Encoded ['A', 'B', 'A']


def test_encode_df_empty():
    """Test encode_df with an empty DataFrame."""
    df = pd.DataFrame({"col_a": [], "col_b": []})
    encoded_tensor = encode_df(df)
    assert encoded_tensor.shape == (0, 2)
    assert isinstance(encoded_tensor, torch.Tensor)
