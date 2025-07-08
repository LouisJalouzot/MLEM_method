import typing as tp

import numpy as np
import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict, Field
from sklearn.datasets import make_spd_matrix
from sklearn.manifold import MDS

from src.dataset import Dataset
from src.utils import BaseModel, seed_from_basemodel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset(path="simulated"))
    level: tp.Literal["simulated"] = "simulated"
    noise_level: float = 0.1
    n_components: int = 768
    _seed: int = None
    _rng: np.random.Generator = None
    _W: np.ndarray = None
    _gt_weights: pd.DataFrame = None

    model_config: ConfigDict = ConfigDict(extra="forbid")
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")

    @property
    def seed(self):
        if self._seed is None:
            self._seed = seed_from_basemodel(self)

        return self._seed

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

        return self._rng

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
        W = make_spd_matrix(n_features, random_state=self.seed)
        W /= np.linalg.norm(W, ord="fro")
        self._W = W
        W_triu = W[*self.dataset.triu_indices]
        self._gt_weights = pd.DataFrame(
            {"Feature": self.dataset.pfeatures, "GTWeight": W_triu}
        )

    @infra.apply
    def forward(self):
        X = self.dataset.encode()

        pX = (X[:, None] - X).abs().clip(0, 1)
        gt_dist = (pX @ self.W * pX).sum(dim=2)
        mds = MDS(
            n_components=self.n_components,
            dissimilarity="precomputed",
            random_state=0,
        )

        repr = mds.fit_transform(gt_dist)
        noise = self.rng.normal(size=repr.shape) * self.noise_level

        return repr + noise

    def __call__(self):
        return torch.from_numpy(self.forward())
