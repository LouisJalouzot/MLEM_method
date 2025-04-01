import torch
from torch import nn
from torch.nn.utils import parametrize
from parametrization_cookbook.torch import MatrixSymPosDef
from src.base_model import BaseModel
import pandas as pd


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


class SPDMatrixLearner(BaseModel):
    def __init__(
        self,
        num_features,
        param="cholesky",
        fro_norm=True,
        init=None,
        init_kwargs={},
        loss="spearman",
        spearman_regularization="l2",
        spearman_regularization_strength=1,
    ):
        super().__init__(
            loss=loss,
            regularization=spearman_regularization,
            regularization_strength=spearman_regularization_strength,
        )
        self.W = nn.Linear(num_features, num_features, bias=False)
        self.triu_indices = torch.triu_indices(num_features, num_features)

        if init is not None:
            getattr(nn.init, init)(self.W.weight, **init_kwargs)

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
            raise ValueError("Invalid parametrization")

        if fro_norm:
            parametrize.register_parametrization(
                self.W, "weight", NormFroParam()
            )

    def get_W(self):
        return self.W.weight

    def get_flat_W(self):
        W = self.get_W()
        W += W.tril(diagonal=-1).T

        return W[*self.triu_indices]

    def get_formatted_W(self, features=None):
        W = self.get_W()
        W += W.triu(1)
        W[*torch.triu_indices(*W.shape, 1)] = torch.nan
        W = pd.DataFrame(W.cpu().detach(), columns=features, index=features)

    def norm_diff(self):
        W = self.get_W()

        return torch.norm(W - W.T).item()

    def min_eigenvalue(self):
        W = self.get_W()
        eigenvalues = torch.linalg.eigvalsh(W)

        return eigenvalues.min().item()

    def check_spd(self):
        print(f"|| W - W^T || = {self.norm_diff():.2g}")
        print(f"Min λ(W) = {self.min_eigenvalue():.2g}")

    def forward(self, X):
        return (self.W(X) * X).sum(dim=1)

    def flat_forward(self, X):
        return X @ self.get_flat_W()
