from __future__ import annotations

import typing as tp

from exca import TaskInfra
from pydantic import ConfigDict, Field

from .dataset import Dataset, SimulatedRepresentations
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import (
    PairwiseDataloaderBuilder,
    PairwiseDataLoaderGenerator,
)
from .sentence_representations import SentenceRepresentations
from .spd_matrix_learner import SPDMatrixLearnerBuilder
from .syntmov2024_representations import SyntMov2024Representations
from .utils import (
    BaseModelSharing,
    get_device,
    seed_everything,
    seed_from_basemodel,
)
from .word_representations import WordRepresentations

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

    from .spd_matrix_learner_torch import SPDMatrixLearner


class Trainer(BaseModelSharing):
    kind: tp.Literal["mlem"] = "mlem"
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    representations: tp.Annotated[
        SentenceRepresentations | WordRepresentations | SimulatedRepresentations | SyntMov2024Representations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())
    dataloader_builder: PairwiseDataloaderBuilder = Field(default_factory=lambda: PairwiseDataloaderBuilder())
    gamma: float = 1
    model_builder: SPDMatrixLearnerBuilder = Field(default_factory=lambda: SPDMatrixLearnerBuilder())
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 1000
    monitor: tp.Literal["grad_norm", "diff_norm", "train_score", "test_score", "loss"] = "loss"
    patience: int = 50
    eps: float = 1e-3

    device: str | None = "cpu"
    unit_indices: list[int] | None = None

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="1")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)
    _shared_fields_config: tp.ClassVar[dict[str, list[str]]] = {
        "dataset": ["estimate_correlations", "representations"],
    }

    def model_post_init(self, __context: tp.Any, /) -> None:
        assert self.dataset.level == self.representations.level, (
            f"Dataset level {self.dataset.level} does not match representations level {self.representations.level}"
        )

    def get_model(self, state_dict=None, device=None) -> SPDMatrixLearner:
        n_features = self.dataset.n_coordinates
        model = self.model_builder.build(n_features=n_features)
        if state_dict is not None:
            model.load_state_dict(state_dict=state_dict)

        return model.to(device or self.device or get_device())

    def get_folds(self, device=None) -> PairwiseDataLoaderGenerator:
        device = device or self.device or get_device()

        # Estimate number of pairs for acceptable variability
        _, n_pairs = self.estimate_correlations.estimate_correlations()

        X = self.dataset.encode()[0].to(device)
        Y = self.representations().to(device)

        # Apply unit selection if specified
        if self.unit_indices is not None:
            Y = Y[:, self.unit_indices]

        return self.dataloader_builder.build(
            X=X,
            Y=Y,
            gamma=self.gamma,
            n_pairs=n_pairs,
            seed=seed_from_basemodel(self),
            signed=self.dataset.mahalanobis,
        )

    @infra.apply(exclude_from_cache_uid=["device"])
    def _train_cached(self) -> tuple[list[torch.Tensor], pd.DataFrame]:
        from .trainer_torch import train

        seed_everything(seed_from_basemodel(self))

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

    def train(self) -> tuple[list[SPDMatrixLearner], pd.DataFrame]:
        all_state_dicts, all_logs = self._train_cached()

        all_models = [self.get_model(state_dict=sd) for sd in all_state_dicts]

        return all_models, all_logs

    def one_log(self) -> pd.DataFrame:
        _, all_logs = self._train_cached()
        return all_logs[0]

    def fi_groups(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(
            {
                "Feature": self.dataset.pcoordinates,
                "Group": self.dataset.pcoordinate_groups,
            }
        )


class OracleTrainer(Trainer):
    kind: tp.Literal["oracle"] = "oracle"

    def get_model(self, state_dict=None, device=None):
        from .simulation import OracleLearner

        device = device or self.device or get_device()
        self.representations()
        Z = self.dataset.encode()[0].to(device)
        _, n_pairs = self.estimate_correlations.estimate_correlations()
        model = OracleLearner(n_features=Z.shape[1])
        model.bind(self.representations.dataset.simulation.transform, Z, n_pairs)
        return model

    def get_folds(self, device=None) -> PairwiseDataLoaderGenerator:
        device = device or self.device or get_device()
        _, n_pairs = self.estimate_correlations.estimate_correlations()
        X = self.dataset.encode()[0].to(device)
        self.representations()
        Y = self.representations.dataset.simulation.transform(X)
        return self.dataloader_builder.build(
            X=X,
            Y=Y,
            gamma=self.gamma,
            n_pairs=n_pairs,
            seed=seed_from_basemodel(self),
            signed=self.dataset.mahalanobis,
        )

    def train(self):
        import pandas as pd

        return [self.get_model()], [pd.DataFrame()]

    def fi_groups(self):
        import pandas as pd

        gser = self.dataset.encode()[1]
        return pd.DataFrame({"Feature": gser.index, "Group": gser.to_numpy()})
