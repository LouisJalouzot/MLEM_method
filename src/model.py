import torch
from torch import nn
from torch.nn.utils import parametrize
from parametrization_cookbook.torch import MatrixSymPosDef
from src.base_model import BaseModel


class SPDExpParam(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X.triu() + X.triu(1).transpose(-1, -2))


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
        param="exp",
        fro_norm=True,
        init_eye=False,
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

        if init_eye:
            nn.init.eye_(self.W.weight)

        if param == "exp":
            parametrize.register_parametrization(
                self.W, "weight", SPDExpParam()
            )
        elif param == "cholesky":
            parametrize.register_parametrization(
                self.W, "weight", CholeskyParam(num_features)
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
