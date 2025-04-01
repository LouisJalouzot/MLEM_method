import torch
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger
from tqdm.auto import tqdm
from pydantic import BaseModel
from exca import TaskInfra
import typing as tp
from src.spd_matrix_learner import SPDMatrixLearnerCfg
from src.pairwise_dataset import PairwiseDatasetCfg
from src.stimulis import Stimulis
from src.text_representations import TextRepresentations
import pandas as pd


class Trainer(BaseModel):
    model: SPDMatrixLearnerCfg = SPDMatrixLearnerCfg()
    dataframe: Stimulis = Stimulis()
    _X: torch.Tensor = None
    representations: TextRepresentations = TextRepresentations()
    _Y: torch.Tensor = None
    dataset: PairwiseDatasetCfg = PairwiseDatasetCfg()
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 500
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    eps: float = 1e-5
    device: str = None
    infra: TaskInfra = TaskInfra(version="1", folder=".cache")

    # Exclude device from caching as it doesn't affect the result
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    def model_post_init(self, __context):
        if self.device is None:
            from src.utils import device

            self.device = device

        self._X = self.dataframe.encode().to(self.device)
        self.model = self.model.build(
            num_features=self.dataframe.num_features
        ).to(self.device)
        self._Y = self.representations.compute_and_combine_representations(
            self.dataframe.stimulis
        ).to(self.device)
        self.dataset = self.dataset.build(self._X, self._Y)

    @infra.apply
    def train(self) -> tp.Tuple[torch.nn.Module, pd.DataFrame]:
        """
        Train a model with caching and optional remote execution.

        Args:
            model: The model to train
            dataloader: DataLoader providing batches

        Returns:
            Trained model
        """
        torch.set_float32_matmul_precision("medium")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            maximize=self.model.maximize,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=("max" if self.model.maximize else "min"),
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        self.model.train()
        prev_w = self.model.get_W().clone()
        start = time()
        logs = []
        pbar = tqdm(
            range(self.max_epochs), desc=f"Training on device {self.device}"
        )
        for i in pbar:
            t = time()

            X_batch, Y_batch = self.dataset[i]
            optimizer.zero_grad(set_to_none=True)
            Y_pred = self.model(X_batch)
            score = self.model.loss(Y_pred, Y_batch)
            score.backward()
            grad_norm = self.model.compute_gradient_norm().item()
            optimizer.step()
            rho = self.model.spearman(Y_pred, Y_batch)
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
            W = self.model.get_W()
            diff_norm = (W - prev_w).norm(p="fro").item()
            log |= {
                "Step Duration": time() - t,
                "Gradient Norm": grad_norm,
                "Diff norm": diff_norm,
            }
            s += " - " + " - ".join([f"{k}: {v:<7.2g}" for k, v in log.items()])
            logger.debug(f"Step {i:<3} / {self.max_epochs} - " + s)
            if diff_norm < self.eps:
                logger.info(
                    f"Convergence reached at step {i} / {self.max_epochs} "
                    f"with diff norm = {diff_norm:.3g} < eps = {self.eps:.3g} "
                    f"after {time() - start:.2g}s"
                )
                break
            prev_w = self.model.get_W().clone()
            if i > self.max_epochs:
                logger.warning(
                    f"Maximum number of epochs reached without convergence: "
                    f"diff norm {diff_norm:.3g} > eps {self.eps:.3g}"
                )
                break

        return self.model.cpu(), pd.DataFrame(logs)
