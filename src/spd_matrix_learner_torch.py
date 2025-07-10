import pandas as pd
import torch
from loguru import logger
from parametrization_cookbook.torch import MatrixSymPosDef
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torchsort import soft_rank


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


class TriuParam(nn.Module):
    def forward(self, W):
        return W.triu()


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
        n_features: int,
        param: str = "cholesky",
        fro_norm: bool = True,
        loss: str = "spearman",
        scoring: str = "spearman",
        spearman_regularization: str = "l2",
        spearman_regularization_strength: float = 1.0,
    ):
        """
        Initialize an SPD Matrix Learner model.

        Args:
            n_features: Number of features in the input
            param: Parametrization type ("exp", "cholesky", "diagonal", "sym", "triu", or "none")
            fro_norm: Whether to apply Frobenius norm normalization
            loss: Loss function to use ("spearman" or "mse")
            scoring: Scoring method to use ("spearman" or "mse")
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
        self.scoring = scoring

        # Create weight matrix
        self.n_features = n_features
        self.W = nn.Linear(n_features, n_features, bias=False, dtype=torch.float32)
        self.triu_indices = torch.triu_indices(n_features, n_features)

        # Add appropriate parametrization

        self.param = param
        match self.param:
            case "exp":
                parametrize.register_parametrization(self.W, "weight", SPDExpParam())
            case "cholesky":
                parametrize.register_parametrization(
                    self.W, "weight", CholeskyParam(n_features)
                )
            case "diagonal":
                parametrize.register_parametrization(self.W, "weight", DiagonalParam())
            case "sym":
                parametrize.register_parametrization(self.W, "weight", SymParam())
            case "triu":
                parametrize.register_parametrization(self.W, "weight", TriuParam())
            case "none":
                pass
            case _:
                raise ValueError(
                    f"Invalid parametrization: {self.param}. Choose from 'exp', 'cholesky', 'diagonal', 'sym', 'triu', or 'none'."
                )

        # Add normalization if requested
        if fro_norm:
            parametrize.register_parametrization(self.W, "weight", NormFroParam())

    def get_W(self) -> torch.Tensor:
        W = self.W.weight.data
        if self.param == "triu":
            W = W.triu()
            W = W + W.T
            W = W - torch.diag(torch.diag(W)) / 2

        return W

    def get_flat_W(self) -> torch.Tensor:
        W = self.get_W()
        W += W.tril(diagonal=-1).T
        W = W[*self.triu_indices]

        return W

    def get_flat_forwatted_W(self, pfeatures) -> pd.DataFrame:
        return pd.DataFrame(
            {"Feature": pfeatures, "Weight": self.get_flat_W().cpu().detach()}
        )

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
                return False
            min_lambda = self.min_eigenvalue()
            if min_lambda <= 0:
                logger.warning(
                    f"Matrix is not positive definite: Min λ(W) {min_lambda:.2g} <= 0"
                )
                return False
            logger.info(
                f"SPD check: Max |W - W^T| = {norm_diff:.2g} - Min λ(W) = {min_lambda:.2g}"
            )
        except Exception as e:
            logger.error(f"SPD check failed: {e}.\n")
            return False
        return True

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

    @torch.no_grad()
    def score(self, x, y):
        if x.shape[1] == self.n_features:
            pred = self.forward(x)
        else:
            pred = self.flat_forward(x)

        if self.scoring == "spearman":
            return self.spearman(pred, y).item()
        elif self.scoring == "mse":
            return self.mse(pred, y).item()
