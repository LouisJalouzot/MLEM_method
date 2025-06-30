import typing as tp
from time import time

import pandas as pd
import torch
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from torch import nn

from src.dataset import Dataset
from src.estimate_correlations import EstimateCorrelations
from src.pairwise_dataloader import (
    PairwiseDataloader,
    PairwiseDataloaderBuilder,
    PairwiseDataLoaderGenerator,
)
from src.sentence_representations import SentenceRepresentations
from src.simulated_representations import SimulatedRepresentations
from src.spd_matrix_learner import SPDMatrixLearner, SPDMatrixLearnerBuilder
from src.utils import BaseModelSharing, get_device
from src.word_representations import WordRepresentations

torch.set_float32_matmul_precision("medium")


def train(
    model: nn.Module,
    train_dataloader: PairwiseDataloader,
    test_dataloader: PairwiseDataloader,
    lr: float = 0.1,
    weight_decay: float = 0,
    max_epochs: int = 500,
    eps: float = 1e-2,
    device: str = "cpu",
    monitor: str = "diff_norm",
    patience: int = 20,
) -> tp.Tuple[nn.Module, pd.DataFrame]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        maximize=model.maximize,
        weight_decay=weight_decay,
        fused=True,
    )
    prev_w = model.get_W().clone()
    start = time()
    logs = []
    converged = False
    logger.debug(f"Training on device {device}.")

    best_score = -torch.inf if model.maximize else torch.inf
    epochs_without_improvement = 0
    best_model_state_dict = None

    for i in range(1, max_epochs + 1):
        t = time()

        X_batch, Y_batch = train_dataloader[i]
        optimizer.zero_grad(set_to_none=True)
        Y_pred = model(X_batch)
        loss = model.loss(Y_pred, Y_batch)
        loss.backward()
        grad_norm = model.compute_gradient_norm()
        optimizer.step()
        X_batch_test, Y_batch_test = test_dataloader[i]
        with torch.no_grad():
            train_score = model.score(X_batch, Y_batch)
            test_score = model.score(X_batch_test, Y_batch_test)

        W = model.get_W()
        diff_norm = (W - prev_w).norm(p=torch.inf).item()
        log = {
            "Batch size": len(X_batch),
            "Loss": loss.item(),
            "Train score": train_score,
            "Test score": test_score,
            "Step Duration": time() - t,
            "Grad norm": grad_norm,
            "Diff norm": diff_norm,
        }
        logs.append(log)
        logger.debug(
            f"Step {i:<3}/{max_epochs} - "
            + " - ".join([f"{k}: {v:<7.2g}" for k, v in log.items()])
        )
        if monitor == "grad_norm" and grad_norm < eps:
            logger.info(
                f"Convergence reached at step {i}/{max_epochs} "
                f"with grad norm={grad_norm:.3g} < eps={eps:.3g} "
                f"after {time() - start:.2g}s"
            )
            converged = True
            break
        if monitor == "diff_norm" and diff_norm < eps:
            logger.info(
                f"Convergence reached at step {i}/{max_epochs} "
                f"with diff norm={diff_norm:.3g} < eps={eps:.3g} "
                f"after {time() - start:.2g}s"
            )
            converged = True
            break
        if monitor in ["train_score", "test_score"]:
            current_score = train_score if monitor == "train_score" else test_score
            improved = (model.maximize and current_score > best_score) or (
                not model.maximize and current_score < best_score
            )

            if improved:
                best_score = current_score
                epochs_without_improvement = 0
                best_model_state_dict = model.state_dict().copy()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping at step {i}/{max_epochs} "
                    f"after {time() - start:.2g}s, "
                    f"best {monitor}={best_score:.3g} "
                    f"did not improve for {patience} epochs"
                )
                converged = True
                if best_model_state_dict is not None:
                    model.load_state_dict(best_model_state_dict)
                break

        prev_w = model.get_W().clone()
        if i >= max_epochs:
            logger.error(
                f"Maximum number of epochs reached without convergence: "
                f"grad norm {grad_norm:.3g} > eps {eps:.3g} and "
                f"diff norm {diff_norm:.3g} > eps {eps:.3g} and "
            )
            break

    model.check_spd()

    logs = pd.DataFrame(logs)
    logs["converged"] = converged

    return model, logs


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
    max_epochs: int = 500
    eps: float = 1e-2
    device: str | None = None
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["estimate_correlations", "representations"],
    }
    monitor: tp.Literal["grad_norm", "diff_norm", "train_score", "test_score"] = (
        "diff_norm"
    )
    patience: int = 100

    def model_post_init(self, __context: tp.Any) -> None:
        assert self.dataset.level == self.representations.level, (
            f"Dataset level {self.dataset.level} does not match "
            f"representations level {self.representations.level}"
        )
        if self.device is None:
            self.device = get_device()

    def get_model(self, state_dict=None, device=None) -> SPDMatrixLearner:
        if device is None:
            device = self.device

        model = self.model_builder.build(n_features=self.dataset.n_features)
        if state_dict is not None:
            model.load_state_dict(state_dict=state_dict)

        return model.to(self.device)

    def get_folds(self) -> PairwiseDataLoaderGenerator:
        # Estimate number of pairs for acceptable variability
        _, n_pairs = self.estimate_correlations.estimate_correlations()

        X = self.dataset.encode().to(self.device)
        Y = self.representations().to(self.device)

        return self.dataloader_builder.build(X=X, Y=Y, gamma=self.gamma, n_pairs=n_pairs)

    @infra.apply(exclude_from_cache_uid=["device"])
    def _train_cached(self) -> tp.Tuple[tp.List[torch.Tensor], pd.DataFrame]:
        # Output state_dict as nn.Module can't be serialized for caching
        all_state_dicts = []
        all_logs = []

        for i, (train_dl, test_dl) in enumerate(self.get_folds()):
            model, logs = train(
                model=self.get_model(),
                train_dataloader=train_dl,
                test_dataloader=test_dl,
                lr=self.lr,
                weight_decay=self.weight_decay,
                max_epochs=self.max_epochs,
                eps=self.eps,
                device=self.device,
                monitor=self.monitor,
                patience=self.patience,
            )
            logs["cv"] = i
            all_state_dicts.append(model.state_dict())
            all_logs.append(logs)

        return all_state_dicts, all_logs

    def train(self) -> tp.Tuple[tp.List[SPDMatrixLearner], pd.DataFrame]:
        all_state_dicts, all_logs = self._train_cached()

        all_models = [
            self.get_model(state_dict=sd, device=self.device) for sd in all_state_dicts
        ]

        return all_models, all_logs
