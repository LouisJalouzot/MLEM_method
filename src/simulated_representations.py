import typing as tp

import numpy as np
import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict, Field

from src.dataset import Dataset
from src.utils import BaseModel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(
        default_factory=lambda: Dataset(path="datasets/simulated.csv")
    )
    level: tp.Literal["simulated"] = "simulated"
    type: tp.Literal["ground-truth", "interaction"] = "ground-truth"
    noise_level: float = 0.1
    scales: tp.List[float] = [0.5, 1, 2]
    interaction_strength: int = 3
    model_config: ConfigDict = ConfigDict(extra="forbid")
    infra: TaskInfra = TaskInfra(folder=".cache")

    def __call__(self):
        df = self.dataset.df_features
        if self.type == "ground-truth":
            repr = pd.DataFrame(
                {
                    "Feat. 1": ["A", "A", "B", "B"],
                    "Feat. 2": ["A", "B", "A", "B"],
                    "Dim. 1": [0.28, -0.12, 0.24, -0.4],
                    "Dim. 2": [-0.05, -0.47, 0.15, 0.27],
                    "Dim. 3": [-0.4, 0.04, 0.4, -0.04],
                }
            )
            repr = df.merge(repr)[["Dim. 1", "Dim. 2", "Dim. 3"]].values
        else:
            repr = np.zeros((len(df), 2))
            repr[:, 1] = 2 * (df["Feat. 1"] == "A") - 1
            interaction = df["Feat. 1"] == df["Feat. 2"]
            repr[:, 0] += self.interaction_strength * (2 * interaction - 1)

        repr += np.random.normal(size=repr.shape, scale=self.scales) * self.noise_level

        return torch.from_numpy(repr)
