from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    import torch

from loguru import logger
from pydantic import ConfigDict

from mlem.utils import BaseModel


class PairwiseDataloader:
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
        seed=None,
    ):
        import torch
        from torch.nn import functional as F

        self.X = X
        self.Y = Y
        if X is not None and Y is not None:
            assert len(X) == len(Y)
            assert X.device == Y.device
        self.n_pairs = n_pairs
        self.gamma = gamma
        if X is not None:
            self.n = len(X)
            self.n_features = X.shape[1]
            self.device = X.device
            self.X = X.float()
        if Y is not None:
            self.n = len(Y)
            self.device = Y.device
            self.Y = Y.float()
        self.max_n_pairs = self.n * (self.n - 1) // 2
        self.nan_to_num = nan_to_num
        self.min = torch.inf
        self.max = -torch.inf
        self.min_max_scale = min_max_scale
        self.logged_debug = False
        self.seed = seed
        self.generator = torch.Generator(
            device=self.device if hasattr(self, "device") else "cpu"
        )
        if seed is not None:
            self.generator.manual_seed(seed)

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
        import torch

        if n_pairs > self.max_n_pairs and not self.logged_debug:
            logger.debug(
                f"Number of pairs requested ({n_pairs}) is greater than the total number of pairs in the data ({self.max_n_pairs})."
            )
            self.logged_debug = True

        n_pairs *= n_trials

        ind_1 = torch.randint(
            0, self.n, (n_pairs,), device=self.device, generator=self.generator
        )
        ind_2 = torch.randint(
            0, self.n, (n_pairs,), device=self.device, generator=self.generator
        )
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
    tp.Tuple[PairwiseDataloader, PairwiseDataloader], None, None
]


class PairwiseDataloaderBuilder(BaseModel):
    cv: int | float | None = None
    distance: str | float | int = 2
    nan_to_num: float = 0
    min_max_scale: bool = True

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, context):
        if isinstance(self.cv, int):
            assert self.cv > 1, "if cv is an int, it needs to be greater than 1"
        elif isinstance(self.cv, float):
            assert 0 < self.cv < 1, "if cv is a float, it needs to be between 0 and 1"

    def build_for_estimation(self, X, seed=None) -> PairwiseDataloader:
        return PairwiseDataloader(
            X=X,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
            seed=seed,
        )

    def build(
        self,
        X=None,
        Y=None,
        n_pairs=None,
        gamma=1,
        seed=None,
    ) -> PairwiseDataLoaderGenerator:
        build_dl = lambda x, y: PairwiseDataloader(
            X=x,
            Y=y,
            n_pairs=n_pairs,
            gamma=gamma,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
            seed=seed,
        )
        assert X is not None or Y is not None, "X or Y must be provided"
        if self.cv is None:
            yield build_dl(X, Y), build_dl(X, Y)
        if isinstance(self.cv, int):
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=0)
            for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
                train_dl = build_dl(X[train_index], Y[train_index])
                test_dl = build_dl(X[test_index], Y[test_index])
                logger.info(f"Split {i} of {self.cv}")
                yield train_dl, test_dl
        elif isinstance(self.cv, float):
            from sklearn.model_selection import train_test_split

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=self.cv, random_state=0
            )

            yield build_dl(X_train, Y_train), build_dl(X_test, Y_test)
