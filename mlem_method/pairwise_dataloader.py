from __future__ import annotations

import typing as tp

import numpy as np
from loguru import logger
from pydantic import ConfigDict, Field

from .utils import BaseModel, corrcoef


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
        elif distance == "correlation":
            self.distance = lambda x, y: 1 - corrcoef(x, y)
        else:
            self.distance = lambda x, y: (x - y).norm(p=distance, dim=-1)

    def pair_delta(self, i, j, X=None):
        X = self.X if X is None else X
        delta = (X[..., i, :] - X[..., j, :]).nan_to_num(self.nan_to_num)
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
    cv: int | float | tuple[list[int], list[int]] | None = None
    n_train: int | None = Field(default=None, ge=2)
    distance: str | float | int = 2
    nan_to_num: float = 0
    min_max_scale: bool = True

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, context):
        if self.n_train is not None and self.cv is None:
            raise ValueError("n_train requires a held-out split (cv)")
        if isinstance(self.cv, tuple):
            assert len(self.cv) == 2 and all(len(i) > 0 for i in self.cv), (
                "cv as a predefined split needs a (train_indices, test_indices) pair"
            )
        elif isinstance(self.cv, int):
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
        """Keep the holdout fixed; subsample only the training pool.

        >>> import torch
        >>> x = torch.arange(20).reshape(-1, 1)
        >>> full, test = next(PairwiseDataloaderBuilder(cv=5).build(X=x, seed=0))
        >>> small, same = next(PairwiseDataloaderBuilder(cv=5, n_train=4).build(X=x, seed=0))
        >>> (small.n, test.n, torch.equal(test.X, same.X))
        (4, 4, True)
        """
        from numpy.random import default_rng
        from sklearn.model_selection import KFold, ShuffleSplit
        from sklearn.utils.validation import check_consistent_length

        assert X is not None or Y is not None, "X or Y must be provided"
        check_consistent_length(X, Y, Y2)
        data = X if X is not None else Y
        if isinstance(self.cv, tuple):
            train_indices, test_indices = (np.asarray(i, dtype=int) for i in self.cv)
            logger.info(f"Predefined split: {len(train_indices)} train / {len(test_indices)} test samples")
            splits = [(train_indices, test_indices)]
        elif self.cv is None:
            splits = [(slice(None), slice(None))]
        elif isinstance(self.cv, int):
            splits = KFold(n_splits=self.cv, shuffle=True, random_state=0).split(data)
        else:
            splits = ShuffleSplit(n_splits=1, test_size=self.cv, random_state=0).split(data)

        for i, (train, test) in enumerate(splits, start=1):
            if self.n_train is not None:
                if self.n_train > len(train):
                    raise ValueError(f"n_train={self.n_train} exceeds the training pool ({len(train)})")
                if self.n_train < len(train):
                    train = default_rng(seed).permutation(train)[:self.n_train]
            if isinstance(self.cv, int):
                logger.info(f"Split {i} of {self.cv}")
            yield tuple(
                PairwiseDataloader(
                    X=X[index] if X is not None else None,
                    Y=Y[index] if Y is not None else None,
                    Y2=Y2[index] if Y2 is not None else None,
                    n_pairs=n_pairs,
                    gamma=gamma,
                    distance=self.distance,
                    nan_to_num=self.nan_to_num,
                    min_max_scale=self.min_max_scale,
                    signed=signed,
                    seed=seed,
                )
                for index in (train, test)
            )

    def get_folds(
        self, X=None, Y=None, Y2=None, n_pairs=None, gamma=1, seed=None, signed=False
    ) -> PairwiseDataLoaderGenerator:
        """Single choke point for fold construction; pair budget and tensors stay caller-side."""
        return self.build(X=X, Y=Y, Y2=Y2, n_pairs=n_pairs, gamma=gamma, seed=seed, signed=signed)
