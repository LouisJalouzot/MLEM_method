import torch
import torch.nn.functional as F
from loguru import logger

# Removed Dataset import
# from torch.utils.data import Dataset
from pydantic import ConfigDict

from src.utils import BaseModel

# Import the core dataset
from src.core.pairwise_dataset import PairwiseDataset

# Removed PairwiseDataset class


class PairwiseDatasetCfg(BaseModel):
    n_pairs: int = None
    gamma: float = 1
    distance: str | float | int = 2
    nan_to_num: float = 0
    min_max_scale: bool = True

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def build(self, X=None, Y=None):
        if self.n_pairs is None:
            n_pairs = 40 * X.shape[1] ** 2
            logger.info(
                f"Number of pairs is not specified. Using estimated {n_pairs} pairs."
            )
        else:
            n_pairs = self.n_pairs

        # Use the imported PairwiseDataset
        return PairwiseDataset(
            X=X,
            Y=Y,
            n_pairs=n_pairs,
            gamma=self.gamma,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
        )
