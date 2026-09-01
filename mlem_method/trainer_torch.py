from copy import deepcopy
from time import time

import pandas as pd
import torch
from loguru import logger
from torch import nn

from .pairwise_dataloader import PairwiseDataloader

torch.set_float32_matmul_precision("medium")
torch.use_deterministic_algorithms(True)


def train(
    model: nn.Module,
    train_dataloader: PairwiseDataloader,
    test_dataloader: PairwiseDataloader,
    lr: float = 0.1,
    weight_decay: float = 0,
    max_epochs: int = 500,
    eps: float = 1e-3,
    device: str = "cpu",
    monitor: str = "loss",
    patience: int = 50,
) -> tuple[nn.Module, pd.DataFrame]:
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

        X_batch, Y_batch, *_ = train_dataloader[i]
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        Y_pred = model(X_batch)
        loss = model.loss(Y_pred, Y_batch)
        loss.backward()
        grad_norm = model.compute_gradient_norm()
        optimizer.step()
        X_batch_test, Y_batch_test, *_ = test_dataloader[i]
        X_batch_test, Y_batch_test = X_batch_test.to(device), Y_batch_test.to(device)
        with torch.no_grad():
            train_score = model.score(X_batch, Y_batch)
            test_score = model.score(X_batch_test, Y_batch_test)

        W = model.get_W()
        diff_norm = (W - prev_w).norm(p=torch.inf).item()
        log = {
            "Step": i,
            "Batch size": len(X_batch),
            "Loss": loss.item(),
            "Train score": train_score,
            "Test score": test_score,
            "Step Duration": time() - t,
            "Grad norm": grad_norm,
            "Diff norm": diff_norm,
        }
        logs.append(log)
        logger.debug(" - ".join([f"{k}: {v:<7.2g}" for k, v in log.items()]))
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
        if monitor in ["train_score", "test_score", "loss"]:
            match monitor:
                case "train_score":
                    current_score = train_score
                case "test_score":
                    current_score = test_score
                case "loss":
                    current_score = loss.item()

            improved = (model.maximize and current_score > best_score) or (
                not model.maximize and current_score < best_score
            )

            if improved:
                best_score = current_score
                epochs_without_improvement = 0
                best_model_state_dict = deepcopy(model.state_dict())
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

    logs = pd.DataFrame(logs)
    logs["converged"] = converged
    logs["spd"] = model.check_spd()

    return model, logs
