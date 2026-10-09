import typing as tp

import numpy as np
from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field, SerializeAsAny, model_validator
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from .baselines import FRRSA, EncodingBaseline, FRRSABaseline
from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloader
from .spd_matrix_learner_torch import SPDMatrixLearner
from .things_dataset import THINGSDataset
from .things_representations import THINGSFmriRepresentations
from .trainer import OracleTrainer, Trainer
from .utils import BaseModelSharing, compute_stats, get_metric, get_n_layers

if tp.TYPE_CHECKING:
    import pandas as pd


def predict_pairs(model, dataloader, X, left, right):
    """Model-agnostic pairwise distances for batched stimulus variants [..., N, F]."""
    import torch

    if isinstance(model, SPDMatrixLearner | FRRSA):  # torch models consume raw pair deltas
        return model(dataloader.pair_delta(left, right, X=X))
    if callable(model):  # Oracle: a simulation transform on tensors
        return dataloader.distance(model(X)[..., left, :], model(X)[..., right, :])
    values = model.predict(X.reshape(-1, X.shape[-1]).cpu().numpy())  # RF: stimuli -> embeddings
    Y = torch.as_tensor(values, device=X.device, dtype=X.dtype).reshape(*X.shape[:-1], -1)
    return dataloader.distance(Y[..., left, :], Y[..., right, :])


def compute_feature_importance(
    model: "SPDMatrixLearner | RandomForestRegressor | FRRSA | tp.Callable",
    dataloader: PairwiseDataloader,
    groups: np.ndarray,
    n_perm: int = 5,
    alpha: float = 0.01,
    scoring: tp.Literal["spearman", "pearson", "mse"] = "spearman",
    perturbations_per_eval: int = 32,
    pfi: tp.Literal["feature", "term"] = "feature",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Score drops from stimulus-feature or MLEM quadratic-term permutation.

    ``groups`` maps input coordinates to theoretical features; categorical blocks
    are always shuffled together. See ``FeatureImportance`` for mode semantics.
    """
    import pandas as pd
    import torch
    from captum.attr import FeatureAblation

    if pfi not in ("feature", "term"):
        raise ValueError(f"Unknown pfi mode: {pfi!r}")
    if pfi == "term" and not isinstance(model, SPDMatrixLearner):
        raise ValueError("pfi='term' requires MLEM")

    metric, maximize = get_metric(scoring)
    sign = 1 if maximize else -1
    X = dataloader.X
    group_ids, names = pd.factorize(groups)
    group_ids = torch.as_tensor(group_ids, device=X.device)
    members = [[name] for name in names]
    if pfi == "term":
        a, b = np.triu_indices(len(names))
        members = [[names[i]] if i == j else [names[i], names[j]] for i, j in zip(a, b)]
        term_ids = torch.empty((len(names), len(names)), dtype=torch.long, device=X.device)
        term_ids[a, b] = term_ids[b, a] = torch.arange(len(members), device=X.device)
        row, col = model.triu_indices
        product_ids = term_ids[group_ids[row], group_ids[col]]
    else:
        blocks = [torch.where(group_ids == k)[0] for k in range(len(names))]
    features = [m[0] if len(m) == 1 else f"({m[0]} x {m[1]})" for m in members]
    keep = X.new_ones(1, len(features))
    effects, scores = [], []

    with torch.no_grad():
        for _ in range(n_perm):
            left, right, delta, observed, *clean_targets = dataloader.sample(dataloader.n_pairs, get_idx=True)
            clean = clean_targets[-1] if clean_targets else observed
            if pfi == "term":
                # Combine both off-diagonal coefficients of the actual forward matrix once.
                W = model.W.weight
                terms = delta[:, row] * delta[:, col] * (W[row, col] + (row != col) * W[col, row])
                contributions = delta.new_zeros(len(delta), len(features)).scatter_add_(
                    1, product_ids.expand(len(delta), -1), terms
                )
                permutation = torch.randperm(len(delta), generator=dataloader.generator, device=X.device)
                change = (contributions[permutation] - contributions).T
                prediction = model(delta)[None]

                def score(mask):
                    return sign * metric(prediction + (1 - mask) @ change, clean)

            else:
                permuted = X.clone()
                for block in blocks:
                    permutation = torch.randperm(dataloader.n, generator=dataloader.generator, device=X.device)
                    permuted[:, block] = X[permutation][:, block]

                def score(mask):
                    # torch.where preserves missing values in the untouched coordinates.
                    variants = torch.where(mask[:, None, group_ids].bool(), X, permuted)
                    return sign * metric(predict_pairs(model, dataloader, variants, left, right), clean)

                prediction = predict_pairs(model, dataloader, X[None], left, right)
            scores.append((sign * metric(prediction, observed)).item())
            attr = FeatureAblation(score).attribute(
                keep, perturbations_per_eval=perturbations_per_eval, show_progress=True
            )[0]
            effects.append(attr.cpu().tolist())

    stats = compute_stats(pd.DataFrame(effects, columns=features), alpha).reset_index(names="Feature")
    importances = pd.DataFrame(
        {
            "Feature": features,
            "AllFeatures": members,
            "Order": ["main" if len(m) == 1 else "interaction" for m in members],
            "Group": features,
        }
    ).merge(stats)
    return importances.sort_values("mean", ascending=False), compute_stats(scores, alpha=alpha).iloc[[0]]


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
    """Permutation importance of fitted pairwise predictions.

    ``pfi="feature"`` (default) shuffles one theoretical feature across stimuli,
    keeping its encoded coordinates together. Works with MLEM, RF, FR-RSA and
    the simulation oracle; returns one main row per feature, with no pairwise
    interaction estimate. These drops include reliance through interactions.

    ``pfi="term"`` requires MLEM. It shuffles one weighted quadratic block
    across comparison rows, keeping the fitted matrix and all other blocks
    fixed. Within-feature blocks are main terms; cross-feature blocks are
    interaction terms. Summing a block before shuffling is equivalent to
    jointly shuffling all its products. Perturbed predictions remain unclipped.

    Both modes average score drops over ``n_perm`` repeats, using the clean
    target when simulations provide one; reported fit scores use observed data.
    In an experiment YAML, set ``base_config.pfi: term`` to opt in.
    Switching PFI modes reuses the same fitted models.
    """

    dataset: SerializeAsAny[Dataset] = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    trainer: tp.Annotated[Trainer | OracleTrainer | EncodingBaseline | FRRSABaseline, Field(discriminator="kind")] = (
        Field(default_factory=lambda: Trainer())
    )

    scoring: tp.Literal["spearman", "pearson", "mse"] = "spearman"
    pfi: tp.Literal["feature", "term"] = "feature"
    n_perm: int = 5
    perturbations_per_eval: int = Field(default=32, ge=1)
    alpha: float = 0.01
    fi_splits: tuple[tp.Literal["train", "test"], ...] = ("test",)

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="18")
    layers_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="9")
    map_infra: MapInfra = MapInfra(version="8")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "infra",
        "layers_infra",
        "map_infra",
        "perturbations_per_eval",
    )
    _shared_fields_config: tp.ClassVar[dict[str, list[str]]] = {
        "dataset": ["trainer", "estimate_correlations"],
        "estimate_correlations": ["trainer"],
    }

    @model_validator(mode="before")
    @classmethod
    def _things_dataset(cls, data: tp.Any) -> tp.Any:
        """THINGS representations need THINGSDataset, which a YAML dict cannot name."""
        if isinstance(data, dict):
            trainer = data.get("trainer")
            representations = trainer.get("representations") if isinstance(trainer, dict) else None
            if isinstance(representations, dict) and str(representations.get("level", "")).startswith("things-"):
                dataset = data.get("dataset", {})
                if isinstance(dataset, dict):
                    data["dataset"] = THINGSDataset(**dataset)
        return data

    @model_validator(mode="after")
    def check_pfi(self):
        if self.pfi == "term" and self.trainer.kind != "mlem":
            raise ValueError("pfi='term' requires MLEM")
        return self

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

        representations = self.trainer.representations
        if isinstance(representations, THINGSFmriRepresentations) and representations.is_empty:
            logger.warning(f"ROI {representations.roi} is empty for sub-{representations.subject}; no results")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        n_features = self.dataset.n_features
        n_effects = n_features if self.pfi == "feature" else n_features * (n_features + 1) // 2
        logger.info(f"Computing {n_effects} {self.pfi} permutation effects with {self.n_perm} permutations.")

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
                weights["step_duration"] = np.nan if logs.empty else logs["Step Duration"].sum()
                weights["estimation_duration"] = np.nan if logs.empty else logs["estimation_duration"].iloc[0]
                weights["training_duration"] = weights["estimation_duration"] + weights["step_duration"]
                weights["n_epochs"] = len(logs)
                gt_weights = getattr(self.trainer.representations, "gt_weights", None)
                if gt_weights is not None:
                    weights = weights.merge(gt_weights)
                    weights["L2"] = np.linalg.norm(weights.GTWeight - weights.Weight)
                all_weights.append(weights)
            elif self.trainer.kind in ("rf", "frrsa"):
                weight = model.feature_importances_ if self.trainer.kind == "rf" else model.w.cpu().numpy()
                weights = pd.DataFrame({"Feature": self.dataset.coordinates, "Weight": weight})
                weights["cv"] = i
                weights["split"] = "train"
                weights["training_duration"] = logs["training_duration"].iloc[0] if len(logs) else np.nan
                weights["n_epochs"] = 1
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
                    perturbations_per_eval=self.perturbations_per_eval,
                    pfi=self.pfi,
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
