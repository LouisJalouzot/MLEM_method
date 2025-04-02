from captum.attr import FeaturePermutation
from tqdm.auto import tqdm
from statsmodels.stats.descriptivestats import describe
from loguru import logger
from src.trainer import Trainer
import torch
from time import time
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from exca import TaskInfra
import typing as tp


def compute_stats(data, alpha=0.01):
    """Compute descriptive statistics with confidence intervals"""
    return describe(
        data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha
    ).T


class FeatureImportance(BaseModel):
    trainer: Trainer

    n_perm: int = 30
    alpha: float = 0.01
    warn_ci: float = 0.01

    infra: TaskInfra = TaskInfra(version="1", folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @infra.apply
    def compute(self) -> tp.Tuple[pd.DataFrame, pd.Series]:
        state_dict, _ = self.trainer.train()
        model, dataset = self.trainer.init(state_dict=state_dict)
        features = self.trainer.features

        n = len(features)
        logger.info(
            f"Computing permutation feature importance with {self.n_perm} permutations "
            f"for {n*(n-1)//2} feature pairs."
        )

        # Create feature pair names
        pfeatures = features[None] + ", " + features[:, None]
        np.fill_diagonal(pfeatures, features)
        pfeatures = pfeatures[*model.triu_indices]

        # Storage for results
        importances = []
        spearman = []
        start = time()

        # Process batches
        with torch.no_grad():
            for i in tqdm(
                range(self.n_perm),
                desc="Computing feature importance",
            ):
                t = time()

                # Sample a batch
                X_batch, Y_batch = dataset[i]

                # Create flattened feature interactions
                X_batch_flat = X_batch[:, None] * X_batch[:, :, None]
                X_batch_flat = X_batch_flat[:, *model.triu_indices]

                # Define the function to measure attributions
                def f(x):
                    pred = model.flat_forward(x)
                    return model.spearman(pred, Y_batch)

                # Calculate feature importance
                feature_perm = FeaturePermutation(f)
                batch_importances = feature_perm.attribute(X_batch_flat).cpu()
                importances.append(batch_importances.double().numpy().squeeze())

                # Calculate baseline performance
                pred = model(X_batch)
                s = model.spearman(pred, Y_batch).item()
                spearman.append(s)

                logger.debug(
                    f"Batch {i:<3} / {self.n_perm} - "
                    f"Duration: {time() - t:.3g}s - "
                    f"Spearman: {s:<8.3g}"
                )
                if i > self.n_perm - 1:
                    break

        # Process results
        importances = np.stack(importances)
        importances = pd.DataFrame(importances, columns=pfeatures)
        importances = compute_stats(importances, alpha=self.alpha)
        importances = importances.sort_values("mean", ascending=False)
        importances = importances.reset_index(names="Feature")

        # Compute spearman statistics
        spearman = compute_stats(spearman, alpha=self.alpha).iloc[0]

        # Log results
        logger.info(
            f"Feature importance computed in {time() - start:.3g}s. "
            f"Mean Spearman = {spearman['mean']:.3g} ± {spearman['std']:.3g}"
        )

        # Warn if there's significant variability
        low, high = spearman["lower_ci"], spearman["upper_ci"]
        if high - low > self.warn_ci:
            logger.warning(
                f"Significant variability between the batches: "
                f"the {(1 - self.alpha)*100:.3g}% confidence interval "
                f"of the Spearman correlation is [{low:.3g}, {high:.3g}] "
                f"which is larger than the threshold {self.warn_ci:.3g}."
            )

        return importances, spearman
