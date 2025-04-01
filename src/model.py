import torch
from torch import nn
from torch.nn.utils import parametrize
from parametrization_cookbook.torch import MatrixSymPosDef
from src.base_model import SPDBaseModel
import pandas as pd
from pydantic import BaseModel
import typing as tp


class DiagonalParam(nn.Module):
    def forward(self, W):
        W = torch.diag(W)
        W = torch.exp(W)
        return torch.diag(W)


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


class SPDMatrixLearnerConfig(BaseModel):
    num_features: int
    param: str = "cholesky"
    fro_norm: bool = True
    init: tp.Optional[str] = None
    init_kwargs: dict = {}
    loss: str = "spearman"
    spearman_regularization: str = "l2"
    spearman_regularization_strength: float = 1.0

    def build(self) -> "SPDMatrixLearner":
        """Build the model using this configuration"""
        return SPDMatrixLearner(
            num_features=self.num_features,
            param=self.param,
            fro_norm=self.fro_norm,
            init=self.init,
            init_kwargs=self.init_kwargs,
            loss=self.loss,
            spearman_regularization=self.spearman_regularization,
            spearman_regularization_strength=self.spearman_regularization_strength,
        )


class SPDMatrixLearner(SPDBaseModel):
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
        super().__init__(
            loss=loss,
            regularization=spearman_regularization,
            regularization_strength=spearman_regularization_strength,
        )

        # Create weight matrix
        self.W = nn.Linear(num_features, num_features, bias=False)
        self.triu_indices = torch.triu_indices(num_features, num_features)

        # Initialize weights if specified
        if init is not None:
            getattr(nn.init, init)(self.W.weight, **init_kwargs)

        # Add appropriate parametrization
        if param == "exp":
            parametrize.register_parametrization(
                self.W, "weight", SPDExpParam()
            )
        elif param == "cholesky":
            parametrize.register_parametrization(
                self.W, "weight", CholeskyParam(num_features)
            )
        elif param == "diagonal":
            parametrize.register_parametrization(
                self.W, "weight", DiagonalParam()
            )
        elif param == "none":
            pass
        else:
            raise ValueError(f"Invalid parametrization: {param}")

        # Add normalization if requested
        if fro_norm:
            parametrize.register_parametrization(
                self.W, "weight", NormFroParam()
            )

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
        return torch.norm(W - W.T).item()

    def min_eigenvalue(self) -> float:
        """Compute the minimum eigenvalue of W"""
        W = self.get_W()
        eigenvalues = torch.linalg.eigvalsh(W)
        return eigenvalues.min().item()

    def check_spd(self) -> None:
        """Print SPD checks"""
        print(f"|| W - W^T || = {self.norm_diff():.2g}")
        print(f"Min λ(W) = {self.min_eigenvalue():.2g}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: weighted sum of transformed features"""
        return (self.W(X) * X).sum(dim=1)

    def flat_forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass using flattened weights"""
        return X @ self.get_flat_W()
