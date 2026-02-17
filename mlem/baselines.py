import typing as tp

import numpy as np
import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from .dataset import Dataset
from .sentence_representations import SentenceRepresentations
from .simulated_representations import SimulatedRepresentations
from .utils import BaseModelSharing, compute_stats
from .word_representations import WordRepresentations


def compute_encoding_baseline(X, Y, n_estimators=10, n_jobs=-2, verbose=False):
    model = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose, random_state=0
    )
    model.fit(X, Y)
    importances = [tree.feature_importances_ for tree in model.estimators_]

    return compute_stats(importances)


class EncodingBaseline(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    representations: (
        tp.Annotated[
            SentenceRepresentations | WordRepresentations | SimulatedRepresentations,
            Field(discriminator="level"),
        ]  # Use sentence or word representations based on the specified level
        | SentenceRepresentations  # Fallback to sentence representations if not specified
    ) = Field(default_factory=lambda: SentenceRepresentations())

    n_estimators: int = 10

    n_jobs: int = -2
    verbose: bool = False
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["representations"]
    }
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("n_jobs", "verbose")

    @infra.apply
    def compute(
        self,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        X = self.dataset.encode()
        Y = self.representations()

        importances = compute_encoding_baseline(
            X, Y, self.n_estimators, self.n_jobs, self.verbose
        )
        importances["Feature"] = self.dataset.features

        return importances.sort_values("mean", ascending=False)


def compute_decoding_baseline(X, Y, n_splits=5):
    all_scores = []
    model = LogisticRegression()
    with tqdm(total=Y.shape[1] * n_splits, desc="Computing decoding baseline") as pbar:
        for i in range(Y.shape[1]):
            y = Y[:, i].int()
            scores = []
            for train, test in StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=0
            ).split(X, y):
                model.fit(X[train], y[train])
                pred_prob = model.predict_proba(X[test])
                if pred_prob.shape[1] == 2:
                    pred_prob = pred_prob[:, 1]
                score = roc_auc_score(
                    y[test], pred_prob, multi_class="ovr", average="weighted"
                )
                scores.append(score)
                pbar.update(1)
            all_scores.append(scores)
    all_scores = pd.DataFrame(all_scores).T
    all_scores = compute_stats(all_scores)

    return all_scores


class DecodingBaseline(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    representations: (
        tp.Annotated[
            SentenceRepresentations | WordRepresentations | SimulatedRepresentations,
            Field(discriminator="level"),
        ]  # Use sentence or word representations based on the specified level
        | SentenceRepresentations  # Fallback to sentence representations if not specified
    ) = Field(default_factory=lambda: SentenceRepresentations())
    n_splits: int = 5

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["representations"]
    }

    @infra.apply
    def compute(
        self,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        X = self.representations()
        Y = self.dataset.encode()
        Y = torch.nan_to_num(Y, nan=-1)

        scores = compute_decoding_baseline(X, Y, self.n_splits)
        scores["Feature"] = self.dataset.features

        return scores.sort_values("mean", ascending=False)
