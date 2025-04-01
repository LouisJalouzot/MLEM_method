from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import BaseModel


class PairwiseDataset(Dataset):
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
        elif Y is not None:
            self.n = len(Y)
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

    def sample(self, n_pairs=None, get_idx=False, only_valid=True):
        if n_pairs > self.max_n_pairs:
            logger.debug(
                f"Number of pairs required ({n_pairs}) is greater than the number of pairs ({self.max_n_pairs})."
            )

        ind_1 = torch.randint(0, self.n, (n_pairs,))
        ind_2 = torch.randint(0, self.n, (n_pairs,))
        if only_valid:
            valid = ind_1 != ind_2
            ind_1 = ind_1[valid]
            ind_2 = ind_2[valid]

        out = tuple()

        if self.X is not None:
            X_1 = self.X[ind_1]
            X_2 = self.X[ind_2]
            X_dist = (X_1 - X_2).nan_to_num(self.nan_to_num).abs().clip(0, 1)
            X_dist = X_dist.reshape(-1, self.n_features)
            out = (X_dist,)

        if self.Y is not None:
            Y_1 = self.Y[ind_1]
            Y_2 = self.Y[ind_2]
            Y_dist = self.distance(Y_1, Y_2).reshape(-1)
            if self.min_max_scale:
                self.min = min(self.min, Y_dist.min())
                self.max = max(self.max, Y_dist.max())
                Y_dist = (Y_dist - self.min) / (self.max - self.min)
            out = (*out, Y_dist)

        if get_idx:
            out = (ind_1, ind_2, *out)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def __getitem__(self, idx):
        return self.sample(int(self.n_pairs * (self.gamma**idx)))


class PairwiseDatasetCfg(BaseModel):
    n_pairs: int = 4096
    gamma: float = 1
    distance: str | float = 2
    nan_to_num: float = 0
    min_max_scale: bool = True

    def build(self, X=None, Y=None):
        return PairwiseDataset(
            X=X,
            Y=Y,
            n_pairs=self.n_pairs,
            gamma=self.gamma,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
        )
