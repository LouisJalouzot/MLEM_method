from __future__ import annotations

import typing as tp

import numpy as np
from pydantic import ConfigDict, Field

from .utils import BaseModel, seed_from_basemodel

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

    from .dataset import Dataset


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


class PolynomialSimulation(BaseModel):
    kind: tp.Literal["poly"] = "poly"
    n: int = 200
    n_cont: int = 8
    cat_sizes: tuple[int, ...] = (2, 2, 2, 3, 4)
    d: int = 300
    p_main: float = 0.5
    p_inter: float = 0.15
    noise: float = 0.2
    outliers: float = 0.02
    outlier_scale: float = 10
    _strength: pd.Series = None
    _Y: torch.Tensor = None
    _powers: np.ndarray = None
    _mean: np.ndarray = None
    _scale: np.ndarray = None
    _A: np.ndarray = None
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def feature_names(self) -> np.ndarray:
        return np.array([*(f"c{k}" for k in range(len(self.cat_sizes))), *(f"x{k}" for k in range(self.n_cont))])

    def make_df(self, seed: int) -> pd.DataFrame:
        import pandas as pd

        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                **{f"c{k}": rng.choice(C, self.n).astype(str) for k, C in enumerate(self.cat_sizes)},
                **{f"x{k}": rng.normal(size=self.n) for k in range(self.n_cont)},
            }
        )

    @property
    def strength(self):
        return self._strength

    def make_Y(self, Z: torch.Tensor, groups: pd.Series, seed: int, signed: bool = False) -> torch.Tensor:
        if self._Y is not None:
            return self._Y

        import pandas as pd
        import torch
        from sklearn.datasets import make_spd_matrix
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler, normalize

        rng = np.random.default_rng(seed)
        coord_feature = groups.to_numpy()

        # Expand encoded coordinates into main terms and pairwise products.
        poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
        H = poly.fit_transform(np.asarray(Z))
        powers = poly.powers_.astype(bool)

        # Map each term to its theoretical feature(s). Drop z_a*z_b when both
        # columns belong to the same feature (two simplex axes of one categorical).
        term_features = [tuple(np.unique(coord_feature[p])) for p in powers]
        keep = np.array([len(f) == p.sum() for f, p in zip(term_features, powers)])
        term_groups = pd.Series([f for f, k in zip(term_features, keep) if k])
        scaler = StandardScaler()
        H = scaler.fit_transform(H[:, keep])

        # Sparse, heavy-tailed strength for each theoretical main effect/interaction.
        unique = term_groups.drop_duplicates()
        prob = np.array([self.p_main if len(g) == 1 else self.p_inter for g in unique])
        strength = pd.Series(
            (rng.random(len(unique)) < prob) * rng.lognormal(size=len(unique)), index=unique, name="strength"
        )

        # Random distributed encoding; normalize for number of coordinates per group.
        scale = term_groups.map(strength / np.sqrt(term_groups.value_counts())).to_numpy()
        A = normalize(rng.normal(size=(H.shape[1], self.d)), axis=1) * scale[:, None]

        Y = H @ A

        # Correlated anisotropic noise, with occasional sample-level outliers.
        eps = rng.multivariate_normal(np.zeros(self.d), make_spd_matrix(self.d, random_state=seed), size=len(Y))
        eps[rng.random(len(Y)) < self.outliers] *= self.outlier_scale
        eps *= self.noise * Y.std() / eps.std()
        self._strength = strength
        self._powers = poly.powers_[keep]
        self._mean = scaler.mean_
        self._scale = scaler.scale_
        self._A = A
        self._Y = torch.from_numpy((Y + eps).astype(np.float32))
        return self._Y

    def transform(self, Z: torch.Tensor) -> torch.Tensor:
        import torch

        if self._A is None:
            raise RuntimeError("call make_Y first")
        Z = torch.as_tensor(Z, dtype=torch.float32)
        powers = torch.as_tensor(self._powers, dtype=Z.dtype, device=Z.device)
        mean = torch.as_tensor(self._mean, dtype=Z.dtype, device=Z.device)
        std = torch.as_tensor(self._scale, dtype=Z.dtype, device=Z.device)
        A = torch.as_tensor(self._A, dtype=Z.dtype, device=Z.device)
        H = Z[:, None, :].pow(powers).prod(2)
        return ((H - mean) / std) @ A


Simulation = tp.Annotated[MdsSimulation | PolynomialSimulation, Field(discriminator="kind")]


class OracleLearner:
    """Dummy model: PFI permutes Z, score is Spearman of G-distances."""

    maximize = True

    def __init__(self, n_features: int, **kwargs):
        import torch

        self.n_features = n_features
        self.triu_indices = torch.triu_indices(n_features, n_features)
        self.transform = None
        self.Z = None
        self.Y = None
        self.n_pairs = 4096
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
