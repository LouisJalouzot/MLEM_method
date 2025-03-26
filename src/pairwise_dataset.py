from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PairwiseDataset(Dataset):
    def __init__(
        self,
        X,
        Y,
        n_pairs=4096,
        gamma=1.1,
        distance=2,
        nan_to_num=0,
        min_max_scale=True,
    ):
        self.X = X
        self.Y = Y
        self.n_pairs = n_pairs
        self.gamma = gamma
        assert len(X) == len(Y)
        self.n = len(X)
        self.max_n_pairs = self.n * (self.n - 1) // 2
        self.n_features = X.shape[1]
        self.nan_to_num = nan_to_num
        self.min = torch.inf
        self.max = -torch.inf
        self.min_max_scale = min_max_scale

        if distance == "cosine":
            self.distance = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)
        else:
            self.distance = lambda x, y: (x - y).norm(p=distance, dim=-1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        batch_n_pairs = int(self.n_pairs * (self.gamma**idx))
        if batch_n_pairs > self.max_n_pairs:
            logger.debug(
                f"Number of pairs required ({batch_n_pairs}) is greater than the number of pairs ({self.max_n_pairs}). Clipping to {self.max_n_pairs}."
            )
            batch_n_pairs = self.max_n_pairs

        ind_1 = torch.randint(0, self.n, (batch_n_pairs,))
        ind_2 = torch.randint(0, self.n, (batch_n_pairs,))

        X_1 = self.X[ind_1]
        X_2 = self.X[ind_2]
        X_dist = (X_1 - X_2).nan_to_num(self.nan_to_num).clip(0, 1)
        X_dist = X_dist.reshape(-1, self.n_features)

        Y_1 = self.Y[ind_1]
        Y_2 = self.Y[ind_2]
        Y_dist = self.distance(Y_1, Y_2).reshape(-1)
        if self.min_max_scale:
            self.min = min(self.min, Y_dist.min())
            self.max = max(self.max, Y_dist.max())
            Y_dist = (Y_dist - self.min) / (self.max - self.min)

        return X_dist, Y_dist
