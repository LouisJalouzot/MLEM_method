import typing as tp
from time import time

import pandas as pd
import torch
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from tqdm_loggable.auto import tqdm

from src.pairwise_dataset import DatasetBuilder
from src.sentence_representations import SentenceRepresentations
from src.spd_matrix_learner import SPDMatrixLearner, SPDMatrixLearnerCfg
from src.stimulis import Stimulis
from src.trainer import train_loop
from src.utils import BaseModel, min_level_debug
from src.word_representations import WordRepresentations


def train_loop(
    model: nn.Module,
    dataset: data.Dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    max_epochs: int,
    eps: float,
    device: str,
) -> tp.Tuple[nn.Module, pd.DataFrame]:
    """Core training loop logic."""
    model.train()
    prev_w = model.get_W().clone()
    start = time()
    logs = []
    pbar = tqdm(
        range(max_epochs),
        desc=f"Training on device {device}",
        disable=min_level_debug(),
    )
    for i in pbar:
        t = time()

        X_batch, Y_batch = dataset[i]
        optimizer.zero_grad(set_to_none=True)
        Y_pred = model(X_batch)
        score = model.loss(Y_pred, Y_batch)
        score.backward()
        grad_norm = model.compute_gradient_norm()
        optimizer.step()
        rho = model.spearman(Y_pred, Y_batch)
        scheduler.step(rho)

        log = {
            "Batch size": len(X_batch),
            "Score": score.item(),
            "Spearman": rho.item(),
            "LR": optimizer.param_groups[0]["lr"],
        }
        logs.append(log)
        s = " - ".join([f"{k}: {v:<8.3g}" for k, v in log.items()])
        pbar.set_postfix_str(s)
        W = model.get_W()
        diff_norm = (W - prev_w).norm(p="fro").item()
        log |= {
            "Step Duration": time() - t,
            "Gradient Norm": grad_norm,
            "Diff norm": diff_norm,
        }
        s += " - " + " - ".join([f"{k}: {v:<7.2g}" for k, v in log.items()])
        logger.debug(f"Step {i:<3} / {max_epochs} - " + s)
        if diff_norm < eps:
            logger.info(
                f"Convergence reached at step {i} / {max_epochs} "
                f"with diff norm = {diff_norm:.3g} < eps = {eps:.3g} "
                f"after {time() - start:.2g}s"
            )
            break
        prev_w = model.get_W().clone()
        if i >= max_epochs - 1:  # Check if max_epochs is reached
            logger.warning(
                f"Maximum number of epochs reached without convergence: "
                f"diff norm {diff_norm:.3g} > eps {eps:.3g}"
            )
            break

    model.check_spd()
    return model, pd.DataFrame(logs)


class Trainer(BaseModel):
    model: SPDMatrixLearnerCfg = SPDMatrixLearnerCfg()
    dataframe: Stimulis = Stimulis()
    representations: SentenceRepresentations | WordRepresentations = Field(
        default=SentenceRepresentations(), discriminator="level"
    )
    dataset: PairwiseDatasetCfg = PairwiseDatasetCfg()
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 500
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    eps: float = 1e-5

    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _device: str | None = None

    @property
    def features(self) -> tp.List[str]:
        return self.dataframe._features

    def init(
        self, state_dict=None
    ) -> tp.Tuple[SPDMatrixLearner, PairwiseDataset]:  # Updated type hints
        torch.set_float32_matmul_precision("medium")
        if self._device is None:
            from src.utils import device

            self._device = device

        Y = self.representations.compute_representations(
            self.dataframe.stimulis
        )
        model = self.model.build(num_features=self.dataframe.num_features)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model = model.to(self._device)
        X = self.dataframe.encode().to(self._device)
        Y = Y.to(self._device)
        dataset = self.dataset.build(X, Y)

        return model, dataset

    @infra.apply
    def train(self) -> tp.Tuple[torch.Tensor, pd.DataFrame]:
        """
        Train a model with caching and optional remote execution.

        Args:
            model: The model to train
            dataloader: DataLoader providing batches

        Returns:
            Trained model state dict and logs
        """
        model, dataset = self.init()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            maximize=model.maximize,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=("max" if model.maximize else "min"),
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        # Call the core training loop
        model, logs = train_loop(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=self.max_epochs,
            eps=self.eps,
            device=self._device,
        )

        # Output state_dict as nn.Module can't be serialized for caching
        return model.state_dict(), logs
