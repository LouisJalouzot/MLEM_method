import typing as tp
from time import time

import pandas as pd
import torch
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from scipy import optimize
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from src.dataset import Stimulis
from src.pairwise_dataloader import PairwiseDataloader, PairwiseDataloaderBuilder
from src.sentence_representations import SentenceRepresentationsCfg
from src.spd_matrix_learner import SPDMatrixLearner, SPDMatrixLearnerBuilder
from src.utils import BaseModel, get_device
from src.word_representations import WordRepresentationsCfg


def train(
    model: nn.Module,
    dataset: PairwiseDataloader,
    lr: float = 0.1,
    weight_decay: float = 0,
    max_epochs: int = 500,
    eps: float = 1e-2,
    device: str = "cpu",
) -> tp.Tuple[nn.Module, pd.DataFrame]:
    """Core training loop logic."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        maximize=model.maximize,
        weight_decay=weight_decay,
    )
    prev_w = model.get_W().clone()
    start = time()
    logs = []
    pbar = tqdm(
        range(1, max_epochs + 1),
        desc=f"Training on device {device}",
        miniters=1,
        disable=True,
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

        log = {
            "Batch size": len(X_batch),
            "Score": score.item(),
            "Spearman": rho.item(),
        }
        s = " - ".join([f"{k}: {v:<8.3g}" for k, v in log.items()])
        pbar.set_postfix_str(s)
        W = model.get_W()
        diff_norm = (W - prev_w).norm(p=torch.inf).item()
        log |= {
            "Step Duration": time() - t,
            "Grad norm": grad_norm,
            "Diff norm": diff_norm,
        }
        logs.append(log)
        logger.debug(
            f"Step {i:<3}/{max_epochs} - "
            + " - ".join([f"{k}: {v:<7.2g}" for k, v in log.items()])
        )
        if grad_norm < eps:
            logger.info(
                f"Convergence reached at step {i}/{max_epochs} "
                f"with grad norm={grad_norm:.3g} < eps={eps:.3g} "
                f"after {time() - start:.2g}s"
            )
            break
        if diff_norm < eps:
            logger.info(
                f"Convergence reached at step {i}/{max_epochs} "
                f"with diff norm={diff_norm:.3g} < eps={eps:.3g} "
                f"after {time() - start:.2g}s"
            )
            break
        prev_w = model.get_W().clone()
        if i >= max_epochs:  # Check if max_epochs is reached
            logger.error(
                f"Maximum number of epochs reached without convergence: "
                f"grad norm {grad_norm:.3g} > eps {eps:.3g} and "
                f"diff norm {diff_norm:.3g} > eps {eps:.3g} and "
            )
            break

    model.check_spd()
    return model, pd.DataFrame(logs)


class Trainer(BaseModel):
    model_builder: SPDMatrixLearnerBuilder = SPDMatrixLearnerBuilder()
    stimulis: Stimulis = Stimulis()
    representations_cfg: WordRepresentationsCfg | SentenceRepresentationsCfg = Field(
        SentenceRepresentationsCfg(), discriminator="level"
    )
    dataset_builder: PairwiseDataloaderBuilder = PairwiseDataloaderBuilder()
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 500
    eps: float = 1e-2
    device: str | None = None
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    def model_post_init(self, __context: tp.Any) -> None:
        if self.device is None:
            self.device = get_device()

    @property
    def features(self) -> tp.List[str]:
        return self.stimulis._features

    def init(self, state_dict=None) -> tp.Tuple[SPDMatrixLearner, PairwiseDataloader]:
        torch.set_float32_matmul_precision("medium")
        if self.device is None:
            self.device = get_device()

        if self.representations_cfg.level == "word":
            Y = self.representations_cfg(
                words=self.stimulis.words,
                sentence_id=self.stimulis.sentence_id,
            )
        elif self.representations_cfg.level == "sentence":
            Y = self.representations_cfg(sentences=self.stimulis.sentences)
        model = self.model_builder.build(num_features=self.stimulis.num_features)
        if state_dict is not None:
            model.load_state_dict(state_dict=state_dict)
        model = model.to(self.device)
        X = self.stimulis.encode().to(self.device)
        Y = Y.to(self.device)
        dataset = self.dataset_builder.build(X, Y)

        return model, dataset

    @infra.apply(exclude_from_cache_uid=["device"])
    def train(self) -> tp.Tuple[torch.Tensor, pd.DataFrame]:
        model, dataset = self.init()
        model, logs = train(
            model=model,
            dataset=dataset,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            eps=self.eps,
            device=self.device,
        )

        # Output state_dict as nn.Module can't be serialized for caching
        return model.state_dict(), logs
