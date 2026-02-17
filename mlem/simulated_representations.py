from __future__ import annotations

import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    import pandas as pd

from pydantic import ConfigDict, Field

from .dataset import Dataset
from .utils import BaseModel, seed_from_basemodel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset(path="simulated"))
    level: tp.Literal["simulated"] = "simulated"
    sparse_spd: bool = False
    sparse_alpha: float = 0.95
    n_components: int = 768
    _W: np.ndarray = None
    _gt_weights: pd.DataFrame = None
    _repr: np.ndarray = None

    model_config: ConfigDict = ConfigDict(extra="forbid")

    @property
    def W(self):
        if self._W is None:
            self.init_weights()

        return self._W

    @property
    def gt_weights(self):
        if self._gt_weights is None:
            self.init_weights()

        return self._gt_weights

    def init_weights(self):
        import pandas as pd
        from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix

        features = self.dataset.features
        n_features = len(features)
        if self.sparse_spd:
            W = make_sparse_spd_matrix(
                n_features,
                alpha=self.sparse_alpha,
                random_state=seed_from_basemodel(self),
            )
        else:
            W = make_spd_matrix(n_features, random_state=seed_from_basemodel(self))
        W /= np.linalg.norm(W, ord="fro")
        self._W = W
        W_triu = W[*self.dataset.triu_indices]
        self._gt_weights = pd.DataFrame(
            {"Feature": self.dataset.pfeatures, "GTWeight": W_triu}
        )

    def __call__(self):
        if self._repr is None:
            import torch
            from sklearn.manifold import MDS

            X = self.dataset.encode()

            pX = (X[:, None] - X).abs().clip(0, 1)
            gt_dist = (pX @ self.W * pX).sum(dim=2)
            mds = MDS(
                n_components=self.n_components,
                dissimilarity="precomputed",
                random_state=0,
            )
            repr = mds.fit_transform(gt_dist)
            scale = repr.std(axis=0)
            rng = np.random.default_rng(seed_from_basemodel(self))
            noise = rng.normal(scale=scale, size=repr.shape)
            noise *= self.dataset.noise_level
            self._repr = torch.from_numpy(repr + noise)

        return self._repr
