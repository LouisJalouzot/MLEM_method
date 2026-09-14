import typing as tp

import numpy as np
from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from .baselines import EncodingBaseline
from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloader
from .simulation import OracleLearner
from .spd_matrix_learner_torch import SPDMatrixLearner
from .trainer import OracleTrainer, Trainer
from .utils import BaseModelSharing, compute_stats, get_metric, get_n_layers

if tp.TYPE_CHECKING:
    import pandas as pd


def compute_feature_importance(
    model: "SPDMatrixLearner | OracleLearner | RandomForestRegressor",
    dataloader: PairwiseDataloader,
    groups: np.ndarray,
    n_perm: int = 5,
    alpha: float = 0.01,
    scoring: tp.Literal["spearman", "pearson", "mse"] = "spearman",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Permutation effects using the same scoring metric for every model."""
    from itertools import combinations

    import pandas as pd
    import torch

    metric, maximize = get_metric(scoring)
    names = list(dict.fromkeys(groups))
    blocks = [torch.as_tensor(np.flatnonzero(groups == name), device=dataloader.device) for name in names]
    pairs = list(combinations(range(len(names)), 2))
    selections = [(), *((k,) for k in range(len(names))), *pairs]
    effects, scores = [], []

    for _ in tqdm(range(n_perm), desc="Permutations"):
        left, right, delta, observed, *clean_targets = dataloader.sample(n_pairs=dataloader.n_pairs, get_idx=True)
        clean = clean_targets[-1] if clean_targets else observed
        permutations = [
            torch.randperm(dataloader.n, generator=dataloader.generator, device=dataloader.device) for _ in names
        ]

        if isinstance(model, RandomForestRegressor):
            observed_predicted = torch.as_tensor(
                model.predict(dataloader.X.cpu().numpy()),
                device=dataloader.device,
                dtype=dataloader.X.dtype,
            ).reshape(dataloader.n, -1)
            observed_distance = (observed_predicted[left] - observed_predicted[right]).norm(dim=-1)
            observed_score = metric(observed_distance, observed)
            clean_scores = []
            for start in range(0, len(selections), 32):
                variants = []
                for selected in selections[start : start + 32]:
                    variant = dataloader.X.clone()
                    for k in selected:
                        variant[:, blocks[k]] = dataloader.X[permutations[k]][:, blocks[k]]
                    variants.append(variant)
                variant_array = torch.stack(variants).cpu().numpy()
                predicted = torch.as_tensor(
                    model.predict(variant_array.reshape(-1, variant_array.shape[-1])),
                    device=dataloader.device,
                    dtype=dataloader.X.dtype,
                ).reshape(len(variants), dataloader.n, -1)
                for batch_predicted in predicted:
                    distance = (batch_predicted[left] - batch_predicted[right]).norm(dim=-1)
                    clean_scores.append(metric(distance, clean))

        elif isinstance(model, SPDMatrixLearner):
            replacements = [
                dataloader.pair_delta(permutations[k][left], permutations[k][right], block)
                for k, block in enumerate(blocks)
            ]
            clean_scores = []
            for selected in selections:
                X = delta.clone()
                for k in selected:
                    X[:, blocks[k]] = replacements[k]
                with torch.no_grad():
                    clean_scores.append(model.score(X, clean, metric))
            with torch.no_grad():
                observed_score = model.score(delta, observed, metric)

        elif isinstance(model, OracleLearner):
            observed_score = model.score_stimuli(dataloader.X, left, right, observed, metric=metric)
            clean_scores = []
            for selected in selections:
                X = dataloader.X.clone()
                for k in selected:
                    X[:, blocks[k]] = dataloader.X[permutations[k]][:, blocks[k]]
                clean_scores.append(model.score_stimuli(X, left, right, clean, metric=metric))

        else:
            observed_score = model.score_stimuli(dataloader.X, left, right, observed)
            clean_scores = []
            for selected in selections:
                X = dataloader.X.clone()
                for k in selected:
                    X[:, blocks[k]] = dataloader.X[permutations[k]][:, blocks[k]]
                clean_scores.append(model.score_stimuli(X, left, right, clean))

        sign = 1 if maximize else -1
        clean_scores = [sign * float(score) for score in clean_scores]
        scores.append(sign * float(observed_score))
        baseline = clean_scores[0]
        main = clean_scores[1 : len(names) + 1]
        joint = clean_scores[len(names) + 1 :]
        effects.append(
            [baseline - score for score in main]
            + [main[a] + main[b] - joint_score - baseline for (a, b), joint_score in zip(pairs, joint)]
        )

    features = [*names, *(f"({names[a]} x {names[b]})" for a, b in pairs)]
    importances = compute_stats(pd.DataFrame(effects, columns=features), alpha).reset_index(names="Feature")
    metadata = pd.DataFrame(
        {
            "Feature": features,
            "AllFeatures": [[name] for name in names] + [[names[a], names[b]] for a, b in pairs],
            "Order": ["main"] * len(names) + ["interaction"] * len(pairs),
        }
    )
    importances = metadata.merge(importances)
    importances["Group"] = importances.Feature
    scores = compute_stats(scores, alpha=alpha).iloc[[0]]
    return importances.sort_values("mean", ascending=False), scores


def compute_cv_stats_per_split(df, alpha=0.01):
    import pandas as pd

    values = "mean" if "mean" in df.columns else "Weight"
    df[values] = df[values].astype(float)
    if "AllFeatures" in df.columns:
        metadata = ["Feature", "AllFeatures"]
        for column in ["Group", "Order"]:
            if column in df.columns:
                metadata.append(column)
        all_features = df[metadata].drop_duplicates("Feature")
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
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    trainer: tp.Annotated[Trainer | OracleTrainer | EncodingBaseline, Field(discriminator="kind")] = Field(
        default_factory=lambda: Trainer()
    )

    scoring: tp.Literal["spearman", "pearson", "mse"] = "spearman"
    n_perm: int = 5
    alpha: float = 0.01
    fi_splits: tuple[tp.Literal["train", "test"], ...] = ("train", "test")

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="12")
    layers_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="3")
    map_infra: MapInfra = MapInfra(version="2")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "infra",
        "layers_infra",
        "map_infra",
    )
    _shared_fields_config: tp.ClassVar[dict[str, list[str]]] = {
        "dataset": ["trainer", "estimate_correlations"],
        "estimate_correlations": ["trainer"],
    }

    @map_infra.apply(item_uid=str, exclude_from_cache_uid=("trainer.representations.layer",))
    def run_layers(
        self, layers: tp.Iterable[int]
    ) -> tp.Iterator[tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]]:
        for layer in layers:
            fi_for_layer = self.infra.clone_obj(trainer={"representations": {"layer": layer}})
            importances, scores, weights = fi_for_layer.compute()
            for df in [importances, scores, weights]:
                df["layer"] = layer
            yield importances, scores, weights

    @layers_infra.apply
    def run_all_layers(
        self,
    ) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        import pandas as pd

        logger.info("Checking that embeddings are cached or launching job")
        self.trainer.representations.precompute()

        model_name = self.trainer.representations.model_name
        n_layers = get_n_layers(model_name)
        layers = range(n_layers + 1)
        logger.info(f"Running feature importance for {len(layers)} layers of model '{model_name}'")
        all_importances, all_scores, all_weights = [], [], []
        for importances, scores, weights in tqdm(self.run_layers(layers), total=len(layers), desc="Layers"):
            all_importances.append(importances)
            all_scores.append(scores)
            all_weights.append(weights)

        return (
            pd.concat(all_importances, ignore_index=True),
            pd.concat(all_scores, ignore_index=True),
            pd.concat(all_weights, ignore_index=True),
        )

    @infra.apply
    def compute(self) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        import pandas as pd

        n_features = self.dataset.n_features
        logger.info(
            f"Computing {n_features + n_features * (n_features - 1) // 2} permutation effects "
            f"with {self.n_perm} permutations."
        )

        all_importances = []
        all_score = []
        all_weights = []

        for i, (model, logs, train_dl, test_dl) in enumerate(self.trainer.train()):
            if self.trainer.kind == "mlem":
                weights = model.get_flat_forwatted_W(pfeatures=self.dataset.pcoordinates)
                weights["cv"] = i
                weights["split"] = "train"
                weights["converged"] = False if logs.empty else logs.converged.iloc[0]
                weights["spd"] = False if logs.empty else logs.spd.iloc[0]
                weights["training_duration"] = 0 if logs.empty else logs["Step Duration"].sum()
                weights["n_epochs"] = len(logs)
                gt_weights = getattr(self.trainer.representations, "gt_weights", None)
                if gt_weights is not None:
                    weights = weights.merge(gt_weights)
                    weights["L2"] = np.linalg.norm(weights.GTWeight - weights.Weight)
                all_weights.append(weights)
            dataloaders = {"train": train_dl, "test": test_dl}
            for split in self.fi_splits:
                importances, score = compute_feature_importance(
                    model,
                    dataloaders[split],
                    self.dataset.coordinate_groups,
                    n_perm=self.n_perm,
                    alpha=self.alpha,
                    scoring=self.scoring,
                )
                for frame in [importances, score]:
                    frame["cv"] = i
                    frame["split"] = split
                all_importances.append(importances)
                all_score.append(score)

        all_importances = pd.concat(all_importances)
        all_score = pd.concat(all_score)
        all_weights = pd.concat(all_weights) if all_weights else pd.DataFrame()

        return all_importances, all_score, all_weights

    def compute_and_aggregate(
        self,
    ) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        all_importances, all_score, all_weights = self.compute()

        if all_importances.cv.nunique() > 1:
            all_importances = compute_cv_stats_per_split(all_importances, alpha=self.alpha)
            all_score = compute_cv_stats_per_split(all_score, alpha=self.alpha)
            if not all_weights.empty:
                all_weights = compute_cv_stats_per_split(all_weights, alpha=self.alpha)

        return all_importances, all_score, all_weights
