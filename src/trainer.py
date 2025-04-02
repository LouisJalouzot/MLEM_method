import torch
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger
from tqdm.auto import tqdm
from pydantic import BaseModel, ConfigDict
from exca import TaskInfra
import typing as tp
from src.spd_matrix_learner import SPDMatrixLearnerCfg
from src.pairwise_dataset import PairwiseDatasetCfg
from src.stimulis import Stimulis
from src.text_representations import TextRepresentations
import pandas as pd


class Trainer(BaseModel):
    model: SPDMatrixLearnerCfg
    dataframe: Stimulis
    representations: TextRepresentations
    dataset: PairwiseDatasetCfg
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
    ) -> tp.Tuple[torch.nn.Module, torch.utils.data.Dataset]:
        torch.set_float32_matmul_precision("medium")
        if self._device is None:
            from src.utils import device

            self._device = device

        Y = self.representations.compute_and_combine_representations(
            self.dataframe.get_stimulis()
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
            Trained model
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
        model.train()
        prev_w = model.get_W().clone()
        start = time()
        logs = []
        pbar = tqdm(
            range(self.max_epochs), desc=f"Training on device {self._device}"
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
            logger.debug(f"Step {i:<3} / {self.max_epochs} - " + s)
            if diff_norm < self.eps:
                logger.info(
                    f"Convergence reached at step {i} / {self.max_epochs} "
                    f"with diff norm = {diff_norm:.3g} < eps = {self.eps:.3g} "
                    f"after {time() - start:.2g}s"
                )
                break
            prev_w = model.get_W().clone()
            if i > self.max_epochs:
                logger.warning(
                    f"Maximum number of epochs reached without convergence: "
                    f"diff norm {diff_norm:.3g} > eps {self.eps:.3g}"
                )
                break

        model.check_spd()

        # Output state_dict as nn.Module can't be serialized for caching
        return model.state_dict(), pd.DataFrame(logs)
