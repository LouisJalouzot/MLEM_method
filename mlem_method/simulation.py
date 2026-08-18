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
        groups = pd.Series([f for f, k in zip(term_features, keep) if k])
        H = StandardScaler().fit_transform(H[:, keep])

        # Sparse, heavy-tailed strength for each theoretical main effect/interaction.
        unique = groups.drop_duplicates()
        prob = np.array([self.p_main if len(g) == 1 else self.p_inter for g in unique])
        strength = pd.Series(
            (rng.random(len(unique)) < prob) * rng.lognormal(size=len(unique)), index=unique, name="strength"
        )

        # Random distributed encoding; normalize for number of coordinates per group.
        scale = groups.map(strength / np.sqrt(groups.value_counts())).to_numpy()
        A = normalize(rng.normal(size=(H.shape[1], self.d)), axis=1) * scale[:, None]

        Y = H @ A

        # Correlated anisotropic noise, with occasional sample-level outliers.
        eps = rng.multivariate_normal(np.zeros(self.d), make_spd_matrix(self.d, random_state=seed), size=len(Y))
        eps[rng.random(len(Y)) < self.outliers] *= self.outlier_scale
        eps *= self.noise * Y.std() / eps.std()
        self._strength = strength
        self._Y = torch.from_numpy((Y + eps).astype(np.float32))
        return self._Y


Simulation = tp.Annotated[MdsSimulation | PolynomialSimulation, Field(discriminator="kind")]


class SimulatedRepresentations(BaseModel):
    dataset: Dataset
    level: tp.Literal["simulated"] = "simulated"
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @property
    def W(self):
        return None if self.dataset.simulation is None else getattr(self.dataset.simulation, "W", None)

    @property
    def gt_weights(self) -> pd.DataFrame | None:
        return None if self.dataset.simulation is None else getattr(self.dataset.simulation, "gt_weights", None)

    def __call__(self) -> torch.Tensor:
        if self.dataset.simulation is None:
            raise ValueError("SimulatedRepresentations requires dataset.simulation")
        Z, groups = self.dataset.encode()
        return self.dataset.simulation.make_Y(
            Z, groups, seed_from_basemodel(self.dataset), signed=self.dataset.mahalanobis
        )
