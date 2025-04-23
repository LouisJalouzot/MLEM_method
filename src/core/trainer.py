import typing as tp
from time import time

import pandas as pd
import torch
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from src.core.pairwise_dataset import PairwiseDataset
from src.core.spd_matrix_learner import SPDMatrixLearner


def train_loop(
    model: SPDMatrixLearner,
    dataset: PairwiseDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    max_epochs: int,
    eps: float,
    device: str,
) -> tp.Tuple[SPDMatrixLearner, pd.DataFrame]:
    """Core training loop logic."""
    model.train()
    prev_w = model.get_W().clone()
    start = time()
    logs = []
    pbar = tqdm(range(max_epochs), desc=f"Training on device {device}")
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
