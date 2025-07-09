import typing as tp

import numpy as np
import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict, Field
from sklearn.datasets import make_spd_matrix
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from src.dataset import Dataset
from src.utils import BaseModel, seed_from_basemodel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset(path="simulated"))
    level: tp.Literal["simulated"] = "simulated"
    noise_level: float = 0.1
    n_components: int = 768
    _W: np.ndarray = None
    _gt_weights: pd.DataFrame = None

    model_config: ConfigDict = ConfigDict(extra="forbid")
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")

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
        W = make_spd_matrix(n_features, random_state=seed_from_basemodel(self))
        W /= np.linalg.norm(W, ord="fro")
        self._W = W
        W_triu = W[*self.dataset.triu_indices]
        self._gt_weights = pd.DataFrame(
            {"Feature": self.dataset.pfeatures, "GTWeight": W_triu}
        )

    @infra.apply(exclude_from_cache_uid=["noise_level"])
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
        repr = StandardScaler().fit_transform(repr)

        return repr

    def __call__(self):
        rng = np.random.default_rng(seed_from_basemodel(self))
        noise = rng.normal(size=self.forward().shape) * self.noise_level

        return torch.from_numpy(self.forward() + noise)
