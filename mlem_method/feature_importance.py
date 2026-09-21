import typing as tp

import numpy as np
from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm

from .baselines import EncodingBaseline, FRRSABaseline
from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloader
from .spd_matrix_learner_torch import SPDMatrixLearner
from .trainer import OracleTrainer, Trainer
from .utils import BaseModelSharing, compute_stats, get_metric, get_n_layers

if tp.TYPE_CHECKING:
    import pandas as pd


def predict_pairs(model, dataloader, X, left, right):
    """Model-agnostic pairwise distances for batched stimulus variants [..., N, F]."""
    import torch

    if isinstance(model, SPDMatrixLearner):
        return model(dataloader.pair_delta(left, right, X=X))
    if isinstance(model, GridSearchCV):
        delta = dataloader.pair_delta(left, right, X=X)
        values = model.predict(delta.square().reshape(-1, delta.shape[-1]).cpu().numpy())
        return torch.as_tensor(values, device=X.device, dtype=X.dtype).reshape(delta.shape[:-1])
    if callable(model):  # Oracle: a simulation transform on tensors
        return dataloader.distance(model(X)[..., left, :], model(X)[..., right, :])
    values = model.predict(X.reshape(-1, X.shape[-1]).cpu().numpy())  # RF: stimuli -> embeddings
    Y = torch.as_tensor(values, device=X.device, dtype=X.dtype).reshape(*X.shape[:-1], -1)
    return dataloader.distance(Y[..., left, :], Y[..., right, :])


def compute_feature_importance(
    model: "SPDMatrixLearner | RandomForestRegressor | GridSearchCV | tp.Callable",
    dataloader: PairwiseDataloader,
    groups: np.ndarray,
    n_perm: int = 5,
    alpha: float = 0.01,
    scoring: tp.Literal["spearman", "pearson", "mse"] = "spearman",
    perturbations_per_eval: int = 32,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Permutation effects using shared stimulus permutations for mains and joints."""
    from itertools import combinations

    import pandas as pd
    import torch
    from captum.attr import FeatureAblation

    metric, maximize = get_metric(scoring)
    sign = 1 if maximize else -1
    X = dataloader.X
    group_ids, names = pd.factorize(groups)
    group_ids = torch.as_tensor(group_ids, device=X.device)
    blocks = [torch.where(group_ids == k)[0] for k in range(len(names))]
    pairs = list(combinations(range(len(names)), 2))
    pair_ids = torch.tensor(pairs, dtype=torch.long, device=X.device)
    # Score columns are mains then joints; joint_map sends each joint to the two groups it switches.
    joint_map = torch.zeros(len(pairs), len(names), device=X.device).scatter_(1, pair_ids, 1)
    keep = X.new_ones(1, len(names) + len(pairs))
    effects, scores = [], []

    with torch.no_grad():
        for _ in range(n_perm):
            left, right, _, observed, *clean_targets = dataloader.sample(dataloader.n_pairs, get_idx=True)
            clean = clean_targets[-1] if clean_targets else observed
            permuted = X.clone()
            for block in blocks:
                permutation = torch.randperm(dataloader.n, generator=dataloader.generator, device=X.device)
                permuted[:, block] = X[permutation][:, block]

            def score(mask):
                # mask 0 switches a selection to its permuted stimuli; torch.where keeps NaNs intact.
                ablated = (1 - mask[:, : len(names)]) + (1 - mask[:, len(names) :]) @ joint_map
                variants = torch.where((ablated == 0)[:, None, group_ids], X, permuted)
                return sign * metric(predict_pairs(model, dataloader, variants, left, right), clean)

            prediction = predict_pairs(model, dataloader, X[None], left, right)
            scores.append((sign * metric(prediction, observed)).item())
            # Captum attr = baseline - score, so main_a = S(all) - S_a and attr for a
            # joint mask gives S(all) - S_ab.
            attr = FeatureAblation(score).attribute(
                keep, perturbations_per_eval=perturbations_per_eval, show_progress=True
            )[0]
            main, joint = attr[: len(names)], attr[len(names) :]
            # inter_ab = S_a + S_b - S_ab - S(all) = joint - main_a - main_b (baseline cancels).
            effects.append(torch.cat([main, joint - main[pair_ids].sum(dim=-1)]).cpu().tolist())

    features = [*names, *(f"({names[a]} x {names[b]})" for a, b in pairs)]
    stats = compute_stats(pd.DataFrame(effects, columns=features), alpha).reset_index(names="Feature")
    importances = pd.DataFrame(
        {
            "Feature": features,
            "AllFeatures": [[name] for name in names] + [[names[a], names[b]] for a, b in pairs],
            "Order": ["main"] * len(names) + ["interaction"] * len(pairs),
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
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    trainer: tp.Annotated[Trainer | OracleTrainer | EncodingBaseline | FRRSABaseline, Field(discriminator="kind")] = (
        Field(default_factory=lambda: Trainer())
    )

    scoring: tp.Literal["spearman", "pearson", "mse"] = "spearman"
    n_perm: int = 5
    perturbations_per_eval: int = Field(default=32, ge=1)
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
        "perturbations_per_eval",
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
                    perturbations_per_eval=self.perturbations_per_eval,
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
