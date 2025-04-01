import torch
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger
from tqdm.auto import tqdm
from pydantic import BaseModel
from exca import TaskInfra
import typing as tp


class TrainingConfig(BaseModel):
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 500
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    eps: float = 1e-5
    device: tp.Optional[torch.device] = None
    infra: TaskInfra = TaskInfra(version="1")

    # Exclude device from caching as it doesn't affect the result
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    @infra.apply
    def train(self, model, dataloader) -> torch.nn.Module:
        """
        Train a model with caching and optional remote execution.

        Args:
            model: The model to train
            dataloader: DataLoader providing batches

        Returns:
            Trained model
        """
        torch.set_float32_matmul_precision("medium")

        # Setup device
        if self.device is None:
            from src.utils import device
        else:
            device = self.device

        # Setup model and optimizer
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            maximize=model.maximize,
            weight_decay=self.weight_decay,
        )

        # Setup scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=("max" if model.maximize else "min"),
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        # Training loop
        model.train()
        prev_w = model.get_W().clone()
        start = time()

        for i, (X_batch, Y_batch) in tqdm(
            enumerate(dataloader), total=self.max_epochs, desc="Training"
        ):
            t = time()

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            Y_pred = model(X_batch)
            score = model.loss(Y_pred, Y_batch)

            # Backward pass
            score.backward()
            grad_norm = model.compute_gradient_norm()
            optimizer.step()

            # Metrics
            rho = model.spearman(Y_pred, Y_batch)
            scheduler.step(rho)
            W = model.get_W()
            diff_norm = (W - prev_w).norm(p="fro")

            # Logging
            step_duration = time() - t
            logger.debug(
                f"Step {i:<3} / {self.max_epochs} - "
                f"Batch size {len(X_batch):<7.2g} - "
                f"Train Score: {score.item():<8.3g} - "
                f"Train Spearman: {rho:<8.3g} - "
                f"Train Duration: {step_duration:<7.2g}s - "
                f"Gradient Norm: {grad_norm:<7.2g} - "
                f"Diff norm {diff_norm:<7.2g} - "
                f"Orig param fro {model.W.parametrizations.weight.original.norm(p='fro'):<7.2g}"
            )

            # Check convergence
            if diff_norm < self.eps:
                logger.info(
                    f"Convergence reached at step {i} / {self.max_epochs} "
                    f"with diff norm = {diff_norm:.3g} < eps = {self.eps:.3g} "
                    f"after {time() - start:.2g}s"
                )
                break

            prev_w = model.get_W().clone()

            # Check if max epochs reached
            if i > self.max_epochs:
                logger.warning(
                    f"Maximum number of epochs reached without convergence: "
                    f"diff norm {diff_norm:.3g} > eps {self.eps:.3g}"
                )
                break

        return model
