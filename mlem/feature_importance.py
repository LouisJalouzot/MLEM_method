import typing as tp
from time import time

import numpy as np
from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloader
from .trainer import Trainer
from .utils import BaseModelSharing, compute_stats, get_n_layers

if tp.TYPE_CHECKING:
    import pandas as pd
    from torch import nn


def compute_feature_importance(
    model: "nn.Module",
    dataloader: PairwiseDataloader,
    clusters: "pd.DataFrame",
    n_perm: int = 5,
    monitor: tp.Literal["std", "ci_width"] = "std",
    thresh: float = 0.01,
    alpha: float = 0.01,
) -> tp.Tuple["pd.DataFrame", "pd.Series"]:
    import pandas as pd
    import torch
    from captum.attr import FeaturePermutation

    features = clusters.Feature
    clusters = torch.from_numpy(clusters.Cluster.values)

    # Storage for results
    importances = []
    score = []
    start = time()

    for i in tqdm(
        range(1, n_perm + 1),
        desc="Computing feature importance",
        disable=True,
    ):
        t = time()
        X_batch, Y_batch = dataloader[i]
        clusters = clusters.to(X_batch.device)

        # Create flattened feature interactions
        X_batch_flat = X_batch[:, None] * X_batch[:, :, None]
        X_batch_flat = X_batch_flat[:, *model.triu_indices]

        # Calculate feature importance
        feature_perm = FeaturePermutation(lambda x: model.score(x, Y_batch))
        batch_importances = feature_perm.attribute(
            X_batch_flat, feature_mask=clusters[None]
        ).cpu()
        batch_importances = batch_importances.double().numpy().squeeze()
        if not model.maximize:
            batch_importances *= -1
        importances.append(batch_importances)

        # Calculate baseline performance
        s = model.score(X_batch, Y_batch)
        score.append(s)

        logger.debug(
            f"Batch {i:<3} / {n_perm} - "
            f"Duration: {time() - t:<8.3f}s - "
            f"Score: {s:<8.3g}"
        )

    # Compute importances statistics
    importances = np.stack(importances)
    importances = pd.DataFrame(importances, columns=features)
    importances = compute_stats(importances, alpha=alpha)
    importances = importances.reset_index(names="Feature")
    importances["Cluster"] = clusters.cpu()
    cols = [col for col in importances.columns if col not in ["Cluster", "Feature"]]
    aggregations = {
        "Feature": [
            ("Feature", lambda x: min(x, key=len)),
            ("AllFeatures", list),
        ]
    } | {col: "first" for col in cols}
    importances = importances.groupby("Cluster").agg(aggregations)
    new_columns = [
        col_tuple[1] if col_tuple[0] == "Feature" else col_tuple[0]
        for col_tuple in importances.columns
    ]
    importances.columns = new_columns
    importances = importances.reset_index()

    importances = importances.sort_values("mean", ascending=False)

    # Compute score statistics
    score_stats = compute_stats(score, alpha=alpha).iloc[0]

    # Log results
    logger.info(
        f"Feature importance computed in {time() - start:.3g}s. "
        f"Mean score = {score_stats['mean']:.3g} ± {score_stats['std']:.3g}"
    )

    # Warn if there's significant variability on the score correlation
    # across batches
    if monitor == "ci_width":
        variability = score_stats["upper_ci"] - score_stats["lower_ci"]
        message = f"the width of the {(1 - alpha) * 100:.3g}% confidence interval of the score correlation is {variability:.3g} "
    elif monitor == "std":
        variability = score_stats["std"]
        message = (
            f"the standard deviation of the score correlation is {variability:.3g} "
        )
    if variability > thresh:
        logger.warning(
            "Significant variability between batches: "
            + message
            + f"which is larger than the threshold {thresh:.3g}."
        )

    return importances, score_stats.to_frame().T


def compute_cv_stats_per_split(df, alpha=0.01):
    import pandas as pd

    values = "mean" if "mean" in df.columns else "Weight"
    df[values] = df[values].astype(float)
    if "AllFeatures" in df.columns:
        all_features = df[["Feature", "AllFeatures"]].drop_duplicates("Feature")
    else:
        all_features = None
    if "Feature" in df.columns:
        df_pivot = df.pivot(index=["cv", "split"], columns="Feature", values=values)
        gb = df_pivot.groupby("split")
    else:
        gb = df.groupby("split")["mean"]

    all_stats = []
    for split, group in gb:
        stats = compute_stats(group, alpha)
        stats["split"] = split
        all_stats.append(stats.reset_index())
    all_stats = pd.concat(all_stats, ignore_index=True)

    if all_features is not None:
        all_stats = all_features.merge(all_stats)

    return all_stats.sort_values("mean", ascending=False)


class FeatureImportance(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(
        default_factory=lambda: EstimateCorrelations()
    )
    trainer: Trainer = Field(default_factory=lambda: Trainer())

    n_perm: int = 5
    monitor: tp.Literal["std", "ci_width"] = "std"
    thresh: float = 0.01
    alpha: float = 0.01

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    layers_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    map_infra: MapInfra = MapInfra()
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["trainer", "estimate_correlations"],
        "estimate_correlations": ["trainer"],
    }

    @map_infra.apply(
        item_uid=str, exclude_from_cache_uid=("trainer.representations.layer",)
    )
    def run_layers(
        self, layers: tp.Iterable[int]
    ) -> tp.Iterator[tp.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]]:
        for layer in layers:
            fi_for_layer = self.infra.clone_obj(
                trainer=dict(representations=dict(layer=layer))
            )
            importances, scores, weights = fi_for_layer.compute()
            for df in [importances, scores, weights]:
                df["layer"] = layer
            yield importances, scores, weights

    @layers_infra.apply
    def run_all_layers(
        self,
    ) -> tp.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        import pandas as pd

        logger.info("Checking that embeddings are cached or launching job")
        self.trainer.representations.precompute()

        model_name = self.trainer.representations.model_name
        n_layers = get_n_layers(model_name)
        layers = list(range(1, n_layers + 1))  # 0 is the embedding layer
        logger.info(
            f"Running feature importance for {len(layers)} layers of model '{model_name}'"
        )
        all_importances, all_scores, all_weights = [], [], []
        for importances, scores, weights in tqdm(
            self.run_layers(layers), total=len(layers), desc="Layers"
        ):
            all_importances.append(importances)
            all_scores.append(scores)
            all_weights.append(weights)

        return (
            pd.concat(all_importances, ignore_index=True),
            pd.concat(all_scores, ignore_index=True),
            pd.concat(all_weights, ignore_index=True),
        )

    @infra.apply
    def compute(self) -> tp.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        import pandas as pd

        # Estimate correlations with forced product
        ec_forced_product = self.estimate_correlations.infra.clone_obj(
            dataset=self.dataset, product=True
        )
        clusters = ec_forced_product.cluster_features()

        all_models, all_logs = self.trainer.train()

        logger.info(
            f"Computing permutation feature importance with {self.n_perm} permutations "
            f"for {clusters.Cluster.max() + 1} clusters of feature pairs."
        )

        all_importances = []
        all_score = []
        all_weights = []

        for i, (model, logs, (train_dl, test_dl)) in enumerate(
            zip(all_models, all_logs, self.trainer.get_folds())
        ):
            weights = model.get_flat_forwatted_W(pfeatures=self.dataset.pfeatures)
            weights["cv"] = i
            weights["split"] = "train"
            weights["converged"] = False if logs.empty else logs.converged.iloc[0]
            weights["spd"] = False if logs.empty else logs.spd.iloc[0]
            weights["training_duration"] = (
                0 if logs.empty else logs["Step Duration"].sum()
            )
            weights["n_epochs"] = len(logs)
            if hasattr(self.trainer.representations, "gt_weights"):
                weights = weights.merge(self.trainer.representations.gt_weights)
                weights["L2"] = np.linalg.norm(weights.GTWeight - weights.Weight)
            all_weights.append(weights)
            for dl, split in [(train_dl, "train"), (test_dl, "test")]:
                importances, score = compute_feature_importance(
                    model=model,
                    dataloader=dl,
                    clusters=clusters,
                    n_perm=self.n_perm,
                    monitor=self.monitor,
                    thresh=self.thresh,
                    alpha=self.alpha,
                )
                for e in [importances, score]:
                    e["cv"] = i
                    e["split"] = split
                all_importances.append(importances)
                all_score.append(score)

        all_importances = pd.concat(all_importances)
        all_score = pd.concat(all_score)
        all_weights = pd.concat(all_weights)

        return all_importances, all_score, all_weights

    def compute_and_aggregate(
        self,
    ) -> tp.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        all_importances, all_score, all_weights = self.compute()

        if all_importances.cv.nunique() > 1:
            all_importances = compute_cv_stats_per_split(
                all_importances, alpha=self.alpha
            )
            all_score = compute_cv_stats_per_split(all_score, alpha=self.alpha)
            all_weights = compute_cv_stats_per_split(all_weights, alpha=self.alpha)

        return all_importances, all_score, all_weights
