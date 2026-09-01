from __future__ import annotations

import typing as tp

from loguru import logger
from pydantic import ConfigDict

from .utils import BaseModel


class PairwiseDataloader:
    """
    A dataset that generates pairs of samples from two datasets (X and Y) and computes the distance between them.
    The dataset can be used for training models that learn to predict the distance between samples.
    """

    def __init__(
        self,
        X=None,
        Y=None,
        Y2=None,
        n_pairs=4096,
        gamma=1,
        distance=2,
        nan_to_num=0,
        min_max_scale=True,
        seed=None,
        signed: bool = False,
    ):
        import torch
        from torch.nn import functional as F

        self.X = X
        self.Y = Y
        self.Y2 = Y2
        if X is not None and Y is not None:
            assert len(X) == len(Y)
            assert X.device == Y.device
        if Y is not None and Y2 is not None:
            assert len(Y2) == len(Y)
            assert Y2.device == Y.device
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
        if Y2 is not None:
            self.n = len(Y2)
            self.device = Y2.device
            self.Y2 = Y2.float()
        self.max_n_pairs = self.n * (self.n - 1) // 2
        self.nan_to_num = nan_to_num
        self.min = torch.inf
        self.max = -torch.inf
        self.min2 = torch.inf
        self.max2 = -torch.inf
        self.min_max_scale = min_max_scale
        self.signed = signed
        self.logged_debug = False
        self.seed = seed
        self.generator = torch.Generator(device=self.device if hasattr(self, "device") else "cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

        if distance == "cosine":
            self.distance = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)
        else:
            self.distance = lambda x, y: (x - y).norm(p=distance, dim=-1)

    def pair_delta(self, i, j, coords=None):
        X = self.X if coords is None else self.X[:, coords]
        delta = (X[i] - X[j]).nan_to_num(self.nan_to_num)
        return delta if self.signed else delta.abs().clip(0, 1)

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

        ind_1 = torch.randint(0, self.n, (n_pairs,), device=self.device, generator=self.generator)
        ind_2 = torch.randint(0, self.n, (n_pairs,), device=self.device, generator=self.generator)
        if only_valid:
            valid = ind_1 != ind_2
            ind_1 = ind_1[valid]
            ind_2 = ind_2[valid]

        out = ()

        if self.X is not None:
            X_dist = self.pair_delta(ind_1, ind_2).reshape(n_trials, -1, self.n_features)
            out = (X_dist,)

        if self.Y is not None:
            Y_1 = self.Y[ind_1]
            Y_2 = self.Y[ind_2]
            Y_dist = self.distance(Y_1, Y_2).reshape(-1)
            if self.min_max_scale:
                self.min = min(self.min, Y_dist.min())
                self.max = max(self.max, Y_dist.max())
                Y_dist = (Y_dist - self.min) / (self.max - self.min)
            Y_dist = Y_dist.reshape(n_trials, -1)
            out = (*out, Y_dist)

        if self.Y2 is not None:
            Y2_1 = self.Y2[ind_1]
            Y2_2 = self.Y2[ind_2]
            Y2_dist = self.distance(Y2_1, Y2_2).reshape(-1)
            if self.min_max_scale:
                self.min2 = min(self.min2, Y2_dist.min())
                self.max2 = max(self.max2, Y2_dist.max())
                Y2_dist = (Y2_dist - self.min2) / (self.max2 - self.min2)
            Y2_dist = Y2_dist.reshape(n_trials, -1)
            out = (*out, Y2_dist)

        if n_trials == 1:
            out = tuple(value[0] for value in out)

        if get_idx:
            out = (ind_1, ind_2, *out)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def __getitem__(self, idx):
        return self.sample(int(self.n_pairs * (self.gamma**idx)))


PairwiseDataLoaderGenerator = tp.Generator[tuple[PairwiseDataloader, PairwiseDataloader]]


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

    def build_for_estimation(self, X, seed=None, signed=False) -> PairwiseDataloader:
        return PairwiseDataloader(
            X=X,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
            signed=signed,
            seed=seed,
        )

    def build(
        self,
        X=None,
        Y=None,
        Y2=None,
        n_pairs=None,
        gamma=1,
        seed=None,
        signed=False,
    ) -> PairwiseDataLoaderGenerator:
        build_dl = lambda x, y, y2=None: PairwiseDataloader(
            X=x,
            Y=y,
            Y2=y2,
            n_pairs=n_pairs,
            gamma=gamma,
            distance=self.distance,
            nan_to_num=self.nan_to_num,
            min_max_scale=self.min_max_scale,
            signed=signed,
            seed=seed,
        )
        assert X is not None or Y is not None, "X or Y must be provided"
        if self.cv is None:
            yield build_dl(X, Y, Y2), build_dl(X, Y, Y2)
        if isinstance(self.cv, int):
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=0)
            for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
                train_dl = build_dl(
                    X[train_index] if X is not None else None,
                    Y[train_index] if Y is not None else None,
                    Y2[train_index] if Y2 is not None else None,
                )
                test_dl = build_dl(
                    X[test_index] if X is not None else None,
                    Y[test_index] if Y is not None else None,
                    Y2[test_index] if Y2 is not None else None,
                )
                logger.info(f"Split {i} of {self.cv}")
                yield train_dl, test_dl
        elif isinstance(self.cv, float):
            from sklearn.model_selection import train_test_split

            arrays = [a for a in [X, Y, Y2] if a is not None]
            splits = train_test_split(*arrays, test_size=self.cv, random_state=0)
            it = iter(splits)
            X_train, X_test = (next(it), next(it)) if X is not None else (None, None)
            Y_train, Y_test = (next(it), next(it)) if Y is not None else (None, None)
            Y2_train, Y2_test = (next(it), next(it)) if Y2 is not None else (None, None)

            yield (
                build_dl(X_train, Y_train, Y2_train),
                build_dl(X_test, Y_test, Y2_test),
            )
