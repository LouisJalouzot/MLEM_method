from __future__ import annotations

import typing as tp

from exca import TaskInfra
from pydantic import ConfigDict, Field

from src.dataset import Dataset
from src.estimate_correlations import EstimateCorrelations
from src.pairwise_dataloader import (
    PairwiseDataloaderBuilder,
    PairwiseDataLoaderGenerator,
)
from src.sentence_representations import SentenceRepresentations
from src.simulated_representations import SimulatedRepresentations
from src.spd_matrix_learner import SPDMatrixLearnerBuilder
from src.utils import BaseModelSharing, get_device
from src.word_representations import WordRepresentations

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

    from src.spd_matrix_learner_torch import SPDMatrixLearner


class Trainer(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(
        default_factory=lambda: EstimateCorrelations()
    )
    representations: (
        tp.Annotated[
            SentenceRepresentations | WordRepresentations | SimulatedRepresentations,
            Field(discriminator="level"),
        ]  # Use sentence or word representations based on the specified level
        | SentenceRepresentations  # Fallback to sentence representations if not specified
    ) = Field(default_factory=lambda: SentenceRepresentations())
    dataloader_builder: PairwiseDataloaderBuilder = Field(
        default_factory=lambda: PairwiseDataloaderBuilder()
    )
    gamma: float = 1
    model_builder: SPDMatrixLearnerBuilder = Field(
        default_factory=lambda: SPDMatrixLearnerBuilder()
    )
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 1000
    monitor: tp.Literal[
        "grad_norm", "diff_norm", "train_score", "test_score", "loss"
    ] = "loss"
    patience: int = 50
    eps: float = 1e-3

    device: str | None = "cpu"
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["estimate_correlations", "representations"],
    }

    def model_post_init(self, __context: tp.Any) -> None:
        assert self.dataset.level == self.representations.level, (
            f"Dataset level {self.dataset.level} does not match "
            f"representations level {self.representations.level}"
        )

    def get_model(self, state_dict=None, device=None) -> SPDMatrixLearner:
        model = self.model_builder.build(n_features=self.dataset.n_features)
        if state_dict is not None:
            model.load_state_dict(state_dict=state_dict)

        return model.to(device or self.device or get_device())

    def get_folds(self, device=None) -> PairwiseDataLoaderGenerator:
        device = device or self.device or get_device()

        # Estimate number of pairs for acceptable variability
        _, n_pairs = self.estimate_correlations.estimate_correlations()

        X = self.dataset.encode().to(device)
        Y = self.representations().to(device)

        return self.dataloader_builder.build(
            X=X, Y=Y, gamma=self.gamma, n_pairs=n_pairs
        )

    @infra.apply(exclude_from_cache_uid=["device"])
    def _train_cached(self) -> tp.Tuple[tp.List[torch.Tensor], pd.DataFrame]:
        from src.trainer_torch import train

        # Output state_dict as nn.Module can't be serialized for caching
        all_state_dicts = []
        all_logs = []

        device = self.device or get_device()

        for i, (train_dl, test_dl) in enumerate(self.get_folds(device=device)):
            model, logs = train(
                model=self.get_model(device=device),
                train_dataloader=train_dl,
                test_dataloader=test_dl,
                lr=self.lr,
                weight_decay=self.weight_decay,
                max_epochs=self.max_epochs,
                eps=self.eps,
                device=device,
                monitor=self.monitor,
                patience=self.patience,
            )
            logs["cv"] = i
            all_state_dicts.append(model.state_dict())
            all_logs.append(logs)

        return all_state_dicts, all_logs

    def train(self) -> tp.Tuple[tp.List[SPDMatrixLearner], pd.DataFrame]:
        all_state_dicts, all_logs = self._train_cached()

        all_models = [self.get_model(state_dict=sd) for sd in all_state_dicts]

        return all_models, all_logs

    def one_log(self) -> pd.DataFrame:
        _, all_logs = self._train_cached()
        return all_logs[0]
