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
from src.utils import BaseModel, encode_df, seed_from_basemodel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset(path="simulated"))
    level: tp.Literal["simulated"] = "simulated"
    noise_level: float = 0.1
    n_components: int = 768
    n_features: int = 16
    n_samples: int = 256
    seed: int = 0
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
        W = make_spd_matrix(self.n_features, random_state=seed_from_basemodel(self))
        W /= np.linalg.norm(W, ord="fro")
        self._W = W
        features = np.array([f"Feat. {i+1}" for i in range(self.n_features)], dtype=str)
        triu_indices = np.triu_indices(self.n_features)
        pfeatures = np.array(
            [
                f"({features[i]} x {features[j]})" if i != j else features[i]
                for i, j in zip(*triu_indices)
            ],
            dtype=str,
        )
        W_triu = W[*triu_indices]
        self._gt_weights = pd.DataFrame({"Feature": pfeatures, "GTWeight": W_triu})

    @infra.apply
    def forward(self):
        rng = np.random.default_rng(seed_from_basemodel(self))
        df = pd.DataFrame(
            rng.choice(
                ["A", "B"],
                size=(self.n_samples, self.n_features),
            ),
        )
        X = encode_df(df)

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
