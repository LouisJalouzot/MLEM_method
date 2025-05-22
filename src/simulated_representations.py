import typing as tp

import numpy as np
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
    noise: float = 0.5
    interaction_strength: int = 3
    model_config: ConfigDict = ConfigDict(extra="forbid")
    infra: TaskInfra = TaskInfra(folder=".cache")

    def __call__(self):
        df = self.dataset.df_features
        repr = np.zeros((len(df), 2))
        repr[:, 1] = 2 * (df.Encoded1 == "A") - 1
        interaction = df.Encoded1 == df.Encoded2
        repr[:, 0] += self.interaction_strength * (2 * interaction - 1)
        repr += np.random.normal(size=repr.shape) * self.noise

        return torch.from_numpy(repr)
