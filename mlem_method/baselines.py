import typing as tp
from time import time

import numpy as np
import pandas as pd
import torch
from exca import TaskInfra
from fracridge import FracRidgeRegressor
from pydantic import ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .dataset import Dataset, SimulatedRepresentations
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloaderBuilder
from .sentence_representations import SentenceRepresentations
from .things_representations import THINGSFmriRepresentations, THINGSMegRepresentations
from .utils import BaseModelSharing, compute_stats, get_device
from .word_representations import WordRepresentations


class FRRSA(torch.nn.Module):
    """FR-RSA as a torch module: standardized squared deltas through the fitted ridge weights."""

    def __init__(self, grid: GridSearchCV):
        super().__init__()
        self.grid = grid
        scaler, ridge = grid.best_estimator_
        for name, values in {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "w": np.asarray(ridge.coef_).squeeze(),
            "b": np.ravel(ridge.intercept_).squeeze(),
        }.items():
            self.register_buffer(name, torch.as_tensor(np.asarray(values), dtype=torch.float64))

    def forward(self, delta):
        mean, scale, w = self.mean.to(delta), self.scale.to(delta), self.w.to(delta)
        return ((delta.square() - mean) / scale) @ w + self.b.to(delta)


class EncodingBaseline(BaseModelSharing):
    kind: tp.Literal["rf"] = "rf"
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    representations: tp.Annotated[
        SentenceRepresentations
        | WordRepresentations
        | SimulatedRepresentations
        | THINGSFmriRepresentations
        | THINGSMegRepresentations,
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
        device = get_device()
        X = self.dataset.encode()[0].to(device)
        Y = self.representations().to(device)
        simulation = self.dataset.simulation
        Y2 = simulation.transform(X) if simulation is not None and simulation.kind == "mlp" else None
        return self.dataloader_builder.get_folds(
            X=X, Y=Y, Y2=Y2, n_pairs=n_pairs, seed=self.dataset.seed, signed=self.dataset.mahalanobis
        )

    @train_infra.apply(exclude_from_cache_uid=("n_jobs", "verbose"))
    def _train_cached(self) -> tuple[list[RandomForestRegressor], "pd.DataFrame"]:
        import pandas as pd

        models, durations = [], []
        for i, (train, _) in enumerate(self.get_folds()):
            start = time()
            models.append(
                RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    random_state=self.dataset.seed,
                ).fit(train.X.cpu().numpy(), train.Y.cpu().numpy())
            )
            durations.append({"cv": i, "training_duration": time() - start})
        return models, pd.DataFrame(durations)

    def train(self):
        models, logs = self._train_cached()
        for (k, log), (train, test) in zip(logs.iterrows(), self.get_folds()):
            yield models[k], log.to_frame().T, train, test

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


class FRRSABaseline(EncodingBaseline):
    kind: tp.Literal["frrsa"] = "frrsa"
    # Unconstrained FR-RSA defaults: ViCCo-Group/frrsa, fitting/crossvalidation.py.
    fractions: tuple[float, ...] = tuple(np.linspace(0.05, 1, 20))
    inner_cv: int = 5
    scoring: tp.Literal["pearson", "spearman"] = "pearson"

    train_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="4")

    @train_infra.apply(exclude_from_cache_uid=("n_jobs", "verbose"))
    def _train_cached(self) -> tuple[list["FRRSA"], "pd.DataFrame"]:
        import pandas as pd

        models, durations = [], []
        corr = pearsonr if self.scoring == "pearson" else spearmanr
        for i, (train, _) in enumerate(self.get_folds()):
            left, right, delta, distance, *_ = train.sample(train.n_pairs, get_idx=True, only_valid=True)
            left, right = left.cpu().numpy(), right.cpu().numpy()
            X = delta.square().cpu().numpy()
            y = distance.cpu().numpy()
            cv = [
                (
                    np.flatnonzero(np.isin(left, tr) & np.isin(right, tr)),
                    np.flatnonzero(np.isin(left, te) & np.isin(right, te)),
                )
                for tr, te in KFold(self.inner_cv, shuffle=True, random_state=self.dataset.seed).split(
                    np.arange(train.n)
                )
            ]
            start = time()
            model = GridSearchCV(
                estimator=make_pipeline(StandardScaler(), FracRidgeRegressor(fit_intercept=True, jit=False)),
                param_grid={"fracridgeregressor__fracs": self.fractions},
                cv=cv,
                scoring=lambda model, X, y: corr(y, model.predict(X)).statistic,
                n_jobs=self.n_jobs,
            ).fit(X, y)
            models.append(FRRSA(model))
            durations.append({"cv": i, "training_duration": time() - start})
        return models, pd.DataFrame(durations)


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
