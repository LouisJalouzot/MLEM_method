from captum.attr import FeaturePermutation
from tqdm.auto import tqdm
from statsmodels.stats.descriptivestats import describe
from loguru import logger
import torch
from time import time
import numpy as np
import pandas as pd
from pydantic import BaseModel
from exca import TaskInfra
import typing as tp


def compute_stats(data, alpha=0.01):
    """Compute descriptive statistics with confidence intervals"""
    return describe(
        data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha
    ).T


class FeatureImportanceConfig(BaseModel):
    n_perm: int = 30
    alpha: float = 0.01
    warn_ci: float = 0.01
    infra: TaskInfra = TaskInfra(version="1")

    @infra.apply
    def compute(
        self, model, dataloader, features
    ) -> tp.Tuple[pd.DataFrame, pd.Series]:
        """
        Compute permutation feature importance with caching support.

        Args:
            model: The model to evaluate
            dataloader: Dataloader providing batches of data
            features: Feature names

        Returns:
            Tuple containing feature importance DataFrame and Spearman correlation stats
        """
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
            for i, (X_batch, Y_batch) in tqdm(
                enumerate(dataloader),
                desc="Computing feature importance",
                total=self.n_perm,
            ):
                t = time()

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
