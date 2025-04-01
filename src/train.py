import torch
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger
from tqdm.auto import tqdm


def train(
    model,
    dataloader,
    device=None,
    lr=0.1,
    weight_decay=0,
    max_epochs=500,
    scheduler_factor=0.5,
    scheduler_patience=10,
    eps=1e-5,
):
    torch.set_float32_matmul_precision("medium")

    if device is None:
        from src.utils import device

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        maximize=model.maximize,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=("max" if model.maximize else "min"),
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    model.train()
    prev_w = model.get_W().clone()
    start = time()
    for i, (X_batch, Y_batch) in tqdm(
        enumerate(dataloader), total=500, desc="Training"
    ):
        t = time()
        optimizer.zero_grad(set_to_none=True)
        Y_pred = model(X_batch)
        score = model.loss(Y_pred, Y_batch)
        score.backward()
        grad_norm = model.compute_gradient_norm()
        optimizer.step()
        rho = model.spearman(Y_pred, Y_batch)
        scheduler.step(rho)
        W = model.get_W()
        diff_norm = (W - prev_w).norm(p="fro")
        t_ = time()
        step_duration = t_ - t
        t = t_
        logger.debug(
            f"Step {i:<3} / {max_epochs} - Batch size {len(X_batch):<7.2g} - Train Score: {score.item():<8.3g} - Train Spearman: {rho:<8.3g} - Train Duration: {step_duration:<7.2g}s - Gradient Norm: {grad_norm:<7.2g} - Diff norm {diff_norm:<7.2g} - Orig param fro {model.W.parametrizations.weight.original.norm(p="fro"):<7.2g}"
        )
        if diff_norm < eps:
            logger.info(
                f"Convergence reached at step {i} / {max_epochs} with diff norm = {diff_norm:.3g} < eps = {eps:.3g} after {time() - start:.2g}s"
            )
            break
        prev_w = model.get_W().clone()
        if i > max_epochs:
            logger.warning(
                f"Maximum number of epochs reached without convergence: diff norm {diff_norm:.3g} > eps {eps:.3g}"
            )
            break

    return model
