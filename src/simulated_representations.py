import typing as tp

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict, Field
from sklearn.datasets import make_spd_matrix
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from src.dataset import Dataset
from src.utils import BaseModel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset(path="simulated"))
    level: tp.Literal["simulated"] = "simulated"
    noise_level: float = 0.1
    n_components: int = 768
    seed: int = 0
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
        features = self.dataset.features
        n_features = len(features)
        W = make_spd_matrix(n_features)
        W /= np.linalg.norm(W, ord="fro")
        self._W = W
        W_triu = W[*self.dataset.triu_indices]
        self._gt_weights = pd.DataFrame(
            {"Feature": self.dataset.pfeatures, "GTWeight": W_triu}
        )

    def __call__(self):
        if self._repr is None:
            X = self.dataset.encode()

            pX = (X[:, None] - X).abs().clip(0, 1)
            gt_dist = (pX @ self.W * pX).sum(dim=2)
            mds = MDS(
                n_components=self.n_components,
                dissimilarity="precomputed",
            )
            repr = mds.fit_transform(gt_dist)
            scale = repr.std(axis=0)
            noise = np.random.normal(scale=scale, size=repr.shape) * self.noise_level
            self._repr = torch.from_numpy(repr + noise)

        return self._repr
