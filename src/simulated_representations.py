import typing as tp

import numpy as np
import torch
from exca import TaskInfra
from pydantic import ConfigDict, Field

from src.dataset import Dataset
from src.utils import BaseModel


class SimulatedRepresentations(BaseModel):
    dataset: Dataset = Field(
        default_factory=lambda: Dataset(csv_path="datasets/simulated.csv")
    )
    level: tp.Literal["simulated"] = "simulated"
    noise_level: float = 0.2
    interaction_strength: float = 0.5
    model_config: ConfigDict = ConfigDict(extra="forbid")
    infra: TaskInfra = TaskInfra(folder=".cache")

    def __call__(self):
        df = self.dataset.df_features
        repr = np.zeros((len(df), 2))
        repr[:, 0] += df.Encoded1 == "A"
        repr[:, 1] += df.Encoded2 == "A"
        interaction = df.Encoded1 == df.Encoded2
        repr += (2 * interaction - 1).values[:, None] * self.interaction_strength
        repr += np.random.normal(size=repr.shape) * self.noise_level

        return torch.from_numpy(repr)
