from __future__ import annotations

import typing as tp

import numpy as np
from pydantic import ConfigDict, Field

from .utils import BaseModel

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch


class MdsSimulation(BaseModel):
    kind: tp.Literal["mds"] = "mds"
    n_features: int = 16
    n_samples: int = 256
    noise_level: float = 0.1
    sparse_spd: bool = False
    sparse_alpha: float = 0.95
    n_components: int = 768
    _W: np.ndarray = None
    _gt_weights: pd.DataFrame = None
    _Y: torch.Tensor = None
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def feature_names(self) -> np.ndarray:
        return np.array([f"Feat. {i + 1}" for i in range(self.n_features)])

    def make_df(self, seed: int) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(
            np.random.default_rng(seed).choice(["A", "B"], size=(self.n_samples, self.n_features)),
            columns=self.feature_names(),
        )

    @property
    def W(self):
        return self._W

    @property
    def gt_weights(self):
        return self._gt_weights

    def make_Y(self, Z: torch.Tensor, groups: pd.Series, seed: int, signed: bool = False) -> torch.Tensor:
        if self._Y is not None:
            return self._Y

        import pandas as pd
        import torch
        from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
        from sklearn.manifold import MDS

        names = groups.index.to_numpy()
        n = len(names)
        if self.sparse_spd:
            W = make_sparse_spd_matrix(n, alpha=self.sparse_alpha, random_state=seed)
        else:
            W = make_spd_matrix(n, random_state=seed)
        W /= np.linalg.norm(W, ord="fro")
        ii, jj = np.triu_indices(n)
        pcoordinates = [names[i] if i == j else f"({names[i]} x {names[j]})" for i, j in zip(ii, jj)]
        self._W = W
        self._gt_weights = pd.DataFrame({"Feature": pcoordinates, "GTWeight": W[ii, jj]})

        pZ = Z[:, None] - Z
        # Legacy MLEM uses clipped |ΔZ|; Mahalanobis uses signed ΔZ.
        if not signed:
            pZ = pZ.abs().clip(0, 1)
        gt_dist = (pZ @ W * pZ).sum(dim=2)
        repr = MDS(n_components=self.n_components, dissimilarity="precomputed", random_state=0).fit_transform(gt_dist)
        noise = np.random.default_rng(seed).normal(scale=repr.std(axis=0), size=repr.shape)
        self._Y = torch.from_numpy(repr + noise * self.noise_level)
        return self._Y


class RandomMlpSimulation(BaseModel):
    """Random MLP with no metric in the data-generating process.

    Standard random weights make semantic features exchangeable in expectation
    and increasingly similar in importance as the network widens. Lognormal
    feature gains create a non-degenerate Oracle FI ranking; all encoded
    coordinates of a categorical feature share one gain. All other parameters
    use standard seeded PyTorch initialization.
    """

    kind: tp.Literal["mlp"] = "mlp"
    n: int = Field(default=160, ge=2)
    n_numeric: int = Field(default=4, ge=0)
    category_cardinalities: tuple[int, ...] = (4, 4, 4)
    hidden_dim: int = Field(default=40, ge=1)
    d: int = Field(default=768, ge=1)
    gain_sigma: float = Field(default=0.55, ge=0)
    noise: float = Field(default=0.25, ge=0)
    _model: torch.nn.Module = None
    _Y: torch.Tensor = None
    _Y0: torch.Tensor = None
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def feature_names(self) -> np.ndarray:
        return np.array(
            [
                *(f"c{k}" for k in range(len(self.category_cardinalities))),
                *(f"x{k}" for k in range(self.n_numeric)),
            ]
        )

    def make_df(self, seed: int) -> pd.DataFrame:
        import pandas as pd

        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                **{
                    f"c{k}": rng.choice(cardinality, self.n).astype(str)
                    for k, cardinality in enumerate(self.category_cardinalities)
                },
                **{f"x{k}": rng.normal(size=self.n) for k in range(self.n_numeric)},
            }
        )

    @property
    def Y0(self):
        if self._Y0 is None:
            raise RuntimeError("call make_Y first")
        return self._Y0

    def transform(self, Z: torch.Tensor) -> torch.Tensor:
        import torch

        if self._model is None:
            raise RuntimeError("call make_Y first")
        self._model.to(Z.device)
        with torch.no_grad():
            return self._model(Z) / self.d**0.5

    def make_Y(self, Z: torch.Tensor, groups: pd.Series, seed: int, signed: bool = False) -> torch.Tensor:
        if self._Y is not None:
            return self._Y

        import torch
        from torch import nn

        rng = np.random.default_rng(seed)
        feature_gains = rng.lognormal(sigma=self.gain_sigma, size=groups.nunique())
        feature_gains /= np.sqrt(np.mean(feature_gains**2))
        gains = torch.from_numpy(groups.map(dict(zip(groups.unique(), feature_gains))).to_numpy(np.float32))

        torch.manual_seed(seed)
        self._model = nn.Sequential(
            nn.Linear(Z.shape[1], self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.d),
            nn.Tanh(),
        )
        with torch.no_grad():
            self._model[0].weight.mul_(gains)
        self._model.requires_grad_(False).to(Z.device)
        self._Y0 = self.transform(Z.float()).cpu()

        eps = rng.normal(size=self._Y0.shape).astype(np.float32)
        eps *= self.noise * self._Y0.std(dim=0).numpy()
        self._Y = self._Y0 + torch.from_numpy(eps)
        return self._Y


Simulation = tp.Annotated[MdsSimulation | RandomMlpSimulation, Field(discriminator="kind")]


class OracleLearner:
    """Fixed simulator used to compute clean geometry and permutation effects."""

    maximize = True

    def __init__(self, n_features: int, **kwargs):
        import torch

        self.n_features = n_features
        self.triu_indices = torch.triu_indices(n_features, n_features)
        self.transform = None
        self.Z = None
        self.Y = None
        self.n_pairs = None
        self.i = self.j = self.D = None

    def parameters(self):
        return iter(())

    def bind(self, transform, Z, n_pairs: int):
        self.transform = transform
        self.Z = Z
        self.Y = transform(Z)
        self.n_pairs = n_pairs
        self.resample_pairs()

    def resample_pairs(self):
        import torch

        n = len(self.Z)
        self.i = torch.randint(n, (self.n_pairs,), device=self.Z.device)
        self.j = torch.randint(n, (self.n_pairs,), device=self.Z.device)
        self.D = (self.Y[self.i] - self.Y[self.j]).norm(dim=1)

    def forward_score(self, Zp):
        from .utils import spearman

        Yp = self.transform(Zp)
        return spearman((Yp[self.i] - Yp[self.j]).norm(dim=1), self.D)

    def score_stimuli(self, Z, i, j, target):
        from .utils import spearman

        Y = self.transform(Z)
        return spearman((Y[i] - Y[j]).norm(dim=1), target).item()

    def batch_importance(self, mask):
        from captum.attr import FeaturePermutation

        self.resample_pairs()
        mask = mask.to(self.Z.device)
        attr = FeaturePermutation(self.forward_score).attribute(self.Z, feature_mask=mask[None])
        imp = attr.reshape(-1, attr.shape[-1]).mean(0).cpu().double().numpy()
        return imp, self.forward_score(self.Z).item()

    def score(self, *args, **kwargs):
        return self.forward_score(self.Z).item()

    def get_W(self):
        import torch

        return torch.zeros(self.n_features, self.n_features)

    def get_flat_forwatted_W(self, pfeatures):
        import pandas as pd

        return pd.DataFrame({"Feature": pfeatures, "Weight": 0.0})

    def to(self, device):
        if self.Z is not None:
            self.Z = self.Z.to(device)
            self.Y = self.Y.to(device)
            self.resample_pairs()
        return self

    def load_state_dict(self, state_dict):
        return self

    def state_dict(self):
        return {}
