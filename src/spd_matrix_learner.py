import typing as tp

import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from parametrization_cookbook.torch import MatrixSymPosDef
from pydantic import ConfigDict
from torch import nn
from torch.nn.utils import parametrize
from torchsort import soft_rank

from src.utils import BaseModel


class DiagonalParam(nn.Module):
    def forward(self, W):
        W = torch.diag(W)
        W = torch.exp(W)
        return torch.diag(W)


class SymParam(nn.Module):
    def forward(self, W):
        return W.triu() + W.triu(1).transpose(-1, -2)


class SPDExpParam(nn.Module):
    def forward(self, W):
        return torch.matrix_exp(W.triu() + W.triu(1).transpose(-1, -2))


class CholeskyParam(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.indices = torch.tril_indices(dim, dim)
        self.param = MatrixSymPosDef(dim=dim)

    def forward(self, W):
        return self.param.reals1d_to_params(W[*self.indices])


class NormFroParam(nn.Module):
    def forward(self, X):
        return X / X.norm(p="fro")


class SPDMatrixLearner(nn.Module):
    def __init__(
        self,
        num_features: int,
        param: str = "cholesky",
        fro_norm: bool = True,
        init: tp.Optional[str] = None,
        init_kwargs: dict = {},
        loss: str = "spearman",
        spearman_regularization: str = "l2",
        spearman_regularization_strength: float = 1.0,
    ):
        """
        Initialize an SPD Matrix Learner model.

        Args:
            num_features: Number of features in the input
            param: Parametrization type ("exp", "cholesky", "diagonal", or "none")
            fro_norm: Whether to apply Frobenius norm normalization
            init: Optional initialization function name (from nn.init)
            init_kwargs: Keyword arguments for initialization function
            loss: Loss function to use ("mse" or "spearman")
            spearman_regularization: Type of regularization for Spearman correlation
            spearman_regularization_strength: Strength of the regularization
        """
        super().__init__()
        self.spearman_regularization = spearman_regularization
        self.spearman_regularization_strength = spearman_regularization_strength
        if loss == "mse":
            self.loss = self.mse
            self.maximize = False
        elif loss == "spearman":
            self.loss = self.spearman_diff
            self.maximize = True
        else:
            raise ValueError(f"Invalid loss function {loss}. Choose 'mse' or 'spearman'.")

        # Create weight matrix
        self.W = nn.Linear(num_features, num_features, bias=False)
        self.triu_indices = torch.triu_indices(num_features, num_features)

        # Initialize weights if specified
        if init is not None:
            getattr(nn.init, init)(self.W.weight, **init_kwargs)

        # Add appropriate parametrization
        if param == "exp":
            parametrize.register_parametrization(self.W, "weight", SPDExpParam())
        elif param == "cholesky":
            parametrize.register_parametrization(
                self.W, "weight", CholeskyParam(num_features)
            )
        elif param == "diagonal":
            parametrize.register_parametrization(self.W, "weight", DiagonalParam())
        elif param == "sym":
            parametrize.register_parametrization(self.W, "weight", SymParam())
        elif param == "none":
            pass
        else:
            raise ValueError(f"Invalid parametrization: {param}")

        # Add normalization if requested
        if fro_norm:
            parametrize.register_parametrization(self.W, "weight", NormFroParam())

    def get_W(self) -> torch.Tensor:
        """Get the weight matrix"""
        return self.W.weight

    def get_flat_W(self) -> torch.Tensor:
        """Get flattened weight matrix (upper triangular)"""
        W = self.get_W()
        W += W.tril(diagonal=-1).T
        return W[*self.triu_indices]

    def get_formatted_W(self, features=None) -> pd.DataFrame:
        """Get the weight matrix as a pandas DataFrame with feature names"""
        W = self.get_W()
        W += W.triu(1)
        W[*torch.triu_indices(*W.shape, 1)] = torch.nan
        return pd.DataFrame(W.cpu().detach(), columns=features, index=features)

    def norm_diff(self) -> float:
        """Compute the Frobenius norm of W - W^T"""
        W = self.get_W()
        return torch.norm(W - W.T, p=torch.inf).item()

    def min_eigenvalue(self) -> float:
        """Compute the minimum eigenvalue of W"""
        W = self.get_W()
        eigenvalues = torch.linalg.eigvalsh(W)
        return eigenvalues.min().item()

    def check_spd(self) -> None:
        try:
            norm_diff = self.norm_diff()
            if norm_diff > 1e-5:
                logger.warning(
                    f"Matrix is not symmetric: Max |W - W^T| = {norm_diff:.2g} > 1e-5"
                )
            min_lambda = self.min_eigenvalue()
            if min_lambda <= 0:
                logger.warning(
                    f"Matrix is not positive definite: Min λ(W) {min_lambda:.2g} <= 0"
                )
            logger.info(
                f"SPD check: Max |W - W^T| = {norm_diff:.2g} - Min λ(W) = {min_lambda:.2g}"
            )
        except Exception as e:
            logger.error(f"SPD check failed: {e}.\n")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: weighted sum of transformed features"""
        return (self.W(X) * X).sum(dim=1)

    def flat_forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass using flattened weights"""
        return X @ self.get_flat_W()

    def compute_gradient_norm(self, norm_type=2):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

        return total_norm

    def corrcoef(self, x, y):
        y_n = y - y.mean()
        x_n = x - x.mean()
        y_n = y_n / y_n.norm()
        x_n = x_n / x_n.norm()

        return (y_n * x_n).sum()

    def spearman(self, x, y):
        dtype = x.dtype
        x_rank = x.argsort().argsort().to(dtype)
        y_rank = y.argsort().argsort().to(dtype)

        return self.corrcoef(x_rank, y_rank)

    def spearman_diff(self, x, y):
        n = x.shape[0]
        x_rank = soft_rank(
            x.reshape(1, -1),
            regularization=self.spearman_regularization,
            regularization_strength=self.spearman_regularization_strength,
        )
        y_rank = soft_rank(
            y.reshape(1, -1),
            regularization=self.spearman_regularization,
            regularization_strength=self.spearman_regularization_strength,
        )

        return self.corrcoef(x_rank / n, y_rank / n)

    def mse(self, x, y):
        return F.mse_loss(x, y)


class SPDMatrixLearnerBuilder(BaseModel):
    param: str = "cholesky"
    fro_norm: bool = True
    init: tp.Optional[str] = None
    init_kwargs: dict = {}
    loss: str = "spearman"
    spearman_regularization: str = "l2"
    spearman_regularization_strength: float = 1.0

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def build(self, num_features) -> SPDMatrixLearner:
        """Build the model using this configuration"""
        return SPDMatrixLearner(
            num_features=num_features,
            param=self.param,
            fro_norm=self.fro_norm,
            init=self.init,
            init_kwargs=self.init_kwargs,
            loss=self.loss,
            spearman_regularization=self.spearman_regularization,
            spearman_regularization_strength=self.spearman_regularization_strength,
        )
