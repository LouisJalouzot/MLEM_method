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

from .dataset import Dataset, SimulatedRepresentations
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloaderBuilder
from .sentence_representations import SentenceRepresentations
from .utils import BaseModelSharing, compute_stats
from .word_representations import WordRepresentations


class EncodingBaseline(BaseModelSharing):
    kind: tp.Literal["rf"] = "rf"
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    representations: tp.Annotated[
        SentenceRepresentations | WordRepresentations | SimulatedRepresentations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())

    n_estimators: int = 100
    dataloader_builder: PairwiseDataloaderBuilder = Field(default_factory=lambda: PairwiseDataloaderBuilder(cv=0.2))

    n_jobs: int = -2
    verbose: bool = False
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="5")
    train_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="3")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[dict[str, list[str]]] = {"dataset": ["estimate_correlations", "representations"]}
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("n_jobs", "verbose", "infra", "train_infra")

    def get_folds(self):
        _, n_pairs = self.estimate_correlations.estimate_correlations()
        X = self.dataset.encode()[0]
        Y = self.representations()
        simulation = self.dataset.simulation
        Y2 = simulation.transform(X) if simulation is not None and simulation.kind == "mlp" else None
        return self.dataloader_builder.build(
            X=X,
            Y=Y,
            Y2=Y2,
            n_pairs=n_pairs,
            seed=self.dataset.seed,
            signed=self.dataset.mahalanobis,
        )

    @train_infra.apply(exclude_from_cache_uid=("n_jobs", "verbose"))
    def _train_cached(self) -> list[RandomForestRegressor]:
        return [
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.dataset.seed,
            ).fit(train.X.cpu().numpy(), train.Y.cpu().numpy())
            for train, _ in self.get_folds()
        ]

    def train(self):
        for model, (train, test) in zip(self._train_cached(), self.get_folds()):
            yield model, pd.DataFrame(), train, test

    @infra.apply
    def compute(self):
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.dataset.seed,
        ).fit(self.dataset.encode()[0], self.representations())
        importances = compute_stats([tree.feature_importances_ for tree in forest.estimators_])
        importances["Feature"] = self.dataset.coordinates
        return importances.sort_values("mean", ascending=False)


def compute_decoding_baseline(X, Y, n_splits=5):
    all_scores = []
    model = LogisticRegression()
    with tqdm(total=Y.shape[1] * n_splits, desc="Computing decoding baseline") as pbar:
        for i in range(Y.shape[1]):
            y = torch.unique(Y[:, i], return_inverse=True)[1]
            scores = []
            for train, test in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y):
                model.fit(X[train], y[train])
                pred_prob = model.predict_proba(X[test])
                if pred_prob.shape[1] == 2:
                    pred_prob = pred_prob[:, 1]
                score = roc_auc_score(y[test], pred_prob, multi_class="ovr", average="weighted")
                scores.append(score)
                pbar.update(1)
            all_scores.append(scores)
    all_scores = pd.DataFrame(all_scores).T
    all_scores = compute_stats(all_scores)

    return all_scores


class DecodingBaseline(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    representations: tp.Annotated[
        SentenceRepresentations | WordRepresentations | SimulatedRepresentations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())
    n_splits: int = 5

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[dict[str, list[str]]] = {"dataset": ["representations"]}

    @infra.apply
    def compute(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self.representations()
        Y = self.dataset.encode()[0]
        Y = torch.nan_to_num(Y, nan=-1)

        scores = compute_decoding_baseline(X, Y, self.n_splits)
        scores["Feature"] = self.dataset.coordinates

        return scores.sort_values("mean", ascending=False)
