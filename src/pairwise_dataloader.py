import typing as tp

import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import ConfigDict
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from src.utils import BaseModel


class PairwiseDataloader(Dataset):
    """
    A dataset that generates pairs of samples from two datasets (X and Y) and computes the distance between them.
    The dataset can be used for training models that learn to predict the distance between samples.
    """

    def __init__(
        self,
        X=None,
        Y=None,
        n_pairs=4096,
        gamma=1,
        distance=2,
        nan_to_num=0,
        min_max_scale=True,
    ):
        self.X = X
        self.Y = Y
        if X is not None and Y is not None:
            assert len(X) == len(Y)
        self.n_pairs = n_pairs
        self.gamma = gamma
        if X is not None:
            self.n = len(X)
            self.n_features = X.shape[1]
            self.device = X.device
        elif Y is not None:
            self.n = len(Y)
            self.device = Y.device
        else:
            raise ValueError("Either X or Y must be provided.")
        self.max_n_pairs = self.n * (self.n - 1) // 2
        self.nan_to_num = nan_to_num
        self.min = torch.inf
        self.max = -torch.inf
        self.min_max_scale = min_max_scale

        if distance == "cosine":
            self.distance = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)
        else:
            self.distance = lambda x, y: (x - y).norm(p=distance, dim=-1)

    def get_X_shape(self):
        if self.X is not None:
            return self.X.shape
        else:
            raise ValueError("X is not provided.")

    def get_Y_shape(self):
        if self.Y is not None:
            return self.Y.shape
        else:
            raise ValueError("Y is not provided.")

    def sample(self, n_pairs=4096, n_trials=1, get_idx=False, only_valid=False):
        if n_pairs > self.max_n_pairs:
            logger.debug(
                f"Number of pairs requested ({n_pairs}) is greater than the total number of pairs in the data ({self.max_n_pairs})."
            )

        n_pairs *= n_trials

        ind_1 = torch.randint(0, self.n, (n_pairs,), device=self.device)
        ind_2 = torch.randint(0, self.n, (n_pairs,), device=self.device)
        if only_valid:
            valid = ind_1 != ind_2
            ind_1 = ind_1[valid]
            ind_2 = ind_2[valid]

        out = tuple()

        if self.X is not None:
            X_1 = self.X[ind_1]
            X_2 = self.X[ind_2]
            X_dist = (X_1 - X_2).nan_to_num(self.nan_to_num).abs().clip(0, 1)
            X_dist = X_dist.reshape(n_trials, -1, self.n_features).squeeze()
            out = (X_dist,)

        if self.Y is not None:
            Y_1 = self.Y[ind_1]
            Y_2 = self.Y[ind_2]
            Y_dist = self.distance(Y_1, Y_2).reshape(-1)
            if self.min_max_scale:
                self.min = min(self.min, Y_dist.min())
                self.max = max(self.max, Y_dist.max())
                Y_dist = (Y_dist - self.min) / (self.max - self.min)
            Y_dist = Y_dist.reshape(n_trials, -1).squeeze()
            out = (*out, Y_dist)

        if get_idx:
            out = (ind_1, ind_2, *out)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def __getitem__(self, idx):
        return self.sample(int(self.n_pairs * (self.gamma**idx)))


PairwiseDataLoaderGenerator = tp.Generator[
    tp.Tuple[PairwiseDataloader, PairwiseDataloader]
]


class PairwiseDataloaderBuilder(BaseModel):
    cv: int | None = None
    distance: str | float | int = 2
    nan_to_num: float = 0
    min_max_scale: bool = True

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def build_for_estimation(self, X) -> PairwiseDataloader:
        return PairwiseDataloader(
            X=X,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
        )

    def build(
        self,
        X=None,
        Y=None,
        n_pairs=None,
        gamma=1,
    ) -> PairwiseDataLoaderGenerator:
        build_dl = lambda x, y: PairwiseDataloader(
            X=x,
            Y=y,
            n_pairs=n_pairs,
            gamma=gamma,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
        )
        if self.cv:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=0)
            for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
                train_dl = build_dl(X[train_index], Y[train_index])
                test_dl = build_dl(X[test_index], Y[test_index])
                logger.info(f"Split {i} of {self.cv}")
                yield train_dl, test_dl
        else:
            yield build_dl(X, Y), build_dl(X, Y)
