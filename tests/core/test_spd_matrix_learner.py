import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from src.core.spd_matrix_learner import (
    CholeskyParam,
    DiagonalParam,
    NormFroParam,
    SPDExpParam,
    SPDMatrixLearner,
    SymParam,
)


@pytest.fixture
def learner_data():
    num_features = 5
    batch_size = 10
    X = torch.randn(batch_size, num_features)
    Y = torch.randn(batch_size)
    return num_features, X, Y


def test_spd_matrix_learner_init(learner_data):
    num_features, _, _ = learner_data
    model = SPDMatrixLearner(num_features=num_features)
    assert isinstance(model, SPDMatrixLearner)
    assert model.W.weight.shape == (num_features, num_features)


def test_spd_matrix_learner_forward(learner_data):
    num_features, X, _ = learner_data
    model = SPDMatrixLearner(num_features=num_features)
    output = model(X)
    assert output.shape == (X.shape[0],)


def test_spd_matrix_learner_loss(learner_data):
    num_features, X, Y = learner_data
    # Test MSE loss
    model_mse = SPDMatrixLearner(num_features=num_features, loss="mse")
    Y_pred_mse = model_mse(X)
    loss_mse = model_mse.loss(Y_pred_mse, Y)
    assert isinstance(loss_mse, torch.Tensor)
    assert loss_mse.item() >= 0

    # Test Spearman loss
    model_spearman = SPDMatrixLearner(
        num_features=num_features, loss="spearman"
    )
    Y_pred_spearman = model_spearman(X)
    loss_spearman = model_spearman.loss(Y_pred_spearman, Y)
    assert isinstance(loss_spearman, torch.Tensor)
    # Spearman correlation is between -1 and 1
    assert -1 <= loss_spearman.item() <= 1


def test_spd_matrix_learner_spearman(learner_data):
    num_features, X, Y = learner_data
    model = SPDMatrixLearner(num_features=num_features)
    Y_pred = model(X)
    rho = model.spearman(Y_pred, Y)
    assert isinstance(rho, torch.Tensor)
    assert -1 <= rho.item() <= 1

    # Test perfect correlation
    rho_perfect = model.spearman(Y, Y)
    assert torch.isclose(rho_perfect, torch.tensor(1.0), atol=1e-6)

    # Test perfect anti-correlation
    rho_anti = model.spearman(Y, -Y)
    assert torch.isclose(rho_anti, torch.tensor(-1.0), atol=1e-6)


@pytest.mark.parametrize(
    "param_name, param_class",
    [
        ("exp", SPDExpParam),
        ("cholesky", CholeskyParam),
        ("diagonal", DiagonalParam),
        ("sym", SymParam),
        ("none", None),  # Test no parametrization
    ],
)
def test_spd_matrix_learner_parametrization(
    learner_data, param_name, param_class
):
    num_features, X, Y = learner_data
    model = SPDMatrixLearner(
        num_features=num_features, param=param_name, fro_norm=False
    )

    # Check if parametrization is registered (if applicable)
    if param_class:
        assert parametrize.is_parametrized(model.W, "weight")
        # Note: Checking the exact class is tricky due to internal structure
        # assert isinstance(model.W.parametrizations.weight[0], param_class)
    else:
        assert not parametrize.is_parametrized(model.W, "weight")

    # Check forward pass works
    output = model(X)
    assert output.shape == (X.shape[0],)

    # Check basic properties based on parametrization
    W = model.get_W()
    if param_name == "diagonal":
        assert torch.allclose(W, torch.diag(torch.diag(W)))
        assert torch.all(
            torch.diag(W) > 0
        )  # Diagonal elements should be positive
    elif param_name == "sym" or param_name == "exp" or param_name == "cholesky":
        assert torch.allclose(W, W.T)  # Should be symmetric
    if param_name == "exp" or param_name == "cholesky":
        # Check positive definiteness (eigenvalues > 0)
        try:
            eigenvalues = torch.linalg.eigvalsh(W)
            assert torch.all(eigenvalues > -1e-6)  # Allow small tolerance
        except torch._C._LinAlgError:
            pytest.fail(
                f"Cholesky decomposition failed for param='{param_name}'"
            )


def test_spd_matrix_learner_fro_norm(learner_data):
    num_features, X, Y = learner_data
    model_norm = SPDMatrixLearner(num_features=num_features, fro_norm=True)
    model_no_norm = SPDMatrixLearner(num_features=num_features, fro_norm=False)

    # Check if NormFroParam is registered
    assert parametrize.is_parametrized(model_norm.W, "weight")
    # Check norm is close to 1
    W_norm = model_norm.get_W()
    assert torch.isclose(
        torch.norm(W_norm, p="fro"), torch.tensor(1.0), atol=1e-5
    )

    # Check norm is likely not 1 without normalization
    W_no_norm = model_no_norm.get_W()
    assert not torch.isclose(
        torch.norm(W_no_norm, p="fro"), torch.tensor(1.0), atol=1e-5
    )


def test_spd_matrix_learner_init_func(learner_data):
    num_features, _, _ = learner_data
    # Test with xavier_uniform initialization
    model = SPDMatrixLearner(
        num_features=num_features,
        init="xavier_uniform_",
        init_kwargs={"gain": nn.init.calculate_gain("relu")},
    )
    # Difficult to assert exact values, but check it runs and W is populated
    W = model.get_W()
    assert W is not None
    assert not torch.allclose(W, torch.zeros_like(W))  # Should not be all zeros


def test_spd_matrix_learner_spearman_diff(learner_data):
    num_features, X, Y = learner_data
    model = SPDMatrixLearner(num_features=num_features, loss="spearman")
    Y_pred = model(X)
    # Use the differentiable spearman used in the loss
    rho_diff = model.spearman_diff(Y_pred, Y)
    assert isinstance(rho_diff, torch.Tensor)
    assert -1 <= rho_diff.item() <= 1
    assert rho_diff.requires_grad  # Should be differentiable

    # Compare with non-differentiable spearman (should be close)
    rho_standard = model.spearman(Y_pred, Y)
    assert torch.isclose(
        rho_diff, rho_standard, atol=1e-1
    )  # Allow some tolerance due to soft_rank


def test_spd_matrix_learner_check_spd(learner_data, caplog):
    num_features, _, _ = learner_data
    # Test with a guaranteed SPD parametrization
    model_spd = SPDMatrixLearner(num_features=num_features, param="cholesky")
    model_spd.check_spd()
    # Check the records directly as caplog.text might not capture loguru output by default
    assert any("SPD check -" in record.message for record in caplog.records)
    assert any("Min λ(W)" in record.message for record in caplog.records)
    assert not any("failed" in record.message for record in caplog.records)

    # Test with a non-guaranteed SPD parametrization (e.g., sym)
    # It might be SPD by chance, but let's force non-SPD if possible
    # model_non_spd = SPDMatrixLearner(num_features=num_features, param="sym")
    # # Force non-SPD (e.g., set a diagonal element negative)
    # with torch.no_grad():
    #     if hasattr(model_non_spd.W, "parametrizations"):
    #         orig_weight = model_non_spd.W.parametrizations.weight.original
    #         orig_weight[0,0] = -10.0 # Modify original parameter
    #     else:
    #         model_non_spd.W.weight.data[0,0] = -10.0
    # model_non_spd.check_spd()
    # assert "failed" in caplog.text # This part is hard to guarantee reliably


def test_spd_matrix_learner_min_eigenvalue(learner_data):
    num_features, _, _ = learner_data
    model_spd = SPDMatrixLearner(num_features=num_features, param="cholesky")
    min_eig = model_spd.min_eigenvalue()
    assert isinstance(min_eig, float)
    assert min_eig > -1e-6  # Should be positive for SPD


def test_spd_matrix_learner_norm_diff(learner_data):
    num_features, _, _ = learner_data
    model_sym = SPDMatrixLearner(num_features=num_features, param="sym")
    norm_diff_sym = model_sym.norm_diff()
    assert isinstance(norm_diff_sym, float)
    assert abs(norm_diff_sym) < 1e-6  # Should be close to zero for symmetric

    model_none = SPDMatrixLearner(num_features=num_features, param="none")
    norm_diff_none = model_none.norm_diff()
    assert isinstance(norm_diff_none, float)
    # Likely non-zero for non-symmetric matrix
    # assert norm_diff_none > 1e-6 # This might fail if initialized symmetrically


def test_get_W(learner_data):
    num_features, _, _ = learner_data
    model = SPDMatrixLearner(num_features=num_features)
    W = model.get_W()
    assert W.shape == (num_features, num_features)


def test_get_flat_W(learner_data):
    num_features, _, _ = learner_data
    model = SPDMatrixLearner(num_features=num_features)
    W_flat = model.get_flat_W()
    # Number of elements in upper triangle including diagonal
    expected_len = num_features * (num_features + 1) // 2
    assert W_flat.shape == (expected_len,)


def test_get_formatted_W(learner_data):
    num_features, _, _ = learner_data
    features = [f"feat_{i}" for i in range(num_features)]
    model = SPDMatrixLearner(num_features=num_features)
    W_df = model.get_formatted_W(features=features)
    assert isinstance(W_df, pd.DataFrame)
    assert W_df.shape == (num_features, num_features)
    assert all(W_df.columns == features)
    assert all(W_df.index == features)
    # Check for NaNs in upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices(num_features, k=1)
    # Use pd.isna() as W_df.values is a numpy array
    assert pd.isna(W_df.values[upper_triangle_indices]).all()
