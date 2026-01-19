"""Univariate analysis for MLEM encoding using MapInfra for efficient parallel processing.

Enables independent analysis of each neural unit (~20k units) with caching and parallelism.
"""

import typing as tp

from exca import MapInfra
from loguru import logger
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from mlem.feature_importance import FeatureImportance
from mlem.utils import BaseModelSharing

if tp.TYPE_CHECKING:
    import pandas as pd


class UnivariateAnalysis(BaseModelSharing):
    feature_importance: FeatureImportance = Field(
        default_factory=lambda: FeatureImportance()
    )

    map_infra: MapInfra = MapInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context):
        assert self.feature_importance.trainer.unit_indices is None

    @map_infra.apply(item_uid=str)
    def run_units(
        self, unit_indices: tp.Iterable[int]
    ) -> tp.Iterator[tp.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]]:
        for unit_idx in unit_indices:
            unit_fi = self.feature_importance.infra.clone_obj(
                dataset=self.feature_importance.dataset,
                trainer=dict(unit_indices=[unit_idx]),
            )
            importances, scores, weights = unit_fi.compute()

            for df in [importances, scores, weights]:
                df["unit"] = unit_idx

            yield importances, scores, weights

    def run(self) -> tp.Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        import pandas as pd

        n_units = self.feature_importance.trainer.representations().shape[1]

        logger.info(f"Launching univariate analysis for {n_units} units")

        all_importances, all_scores, all_weights = [], [], []
        for importances, scores, weights in tqdm(
            self.run_units(list(range(n_units))), total=n_units
        ):
            all_importances.append(importances)
            all_scores.append(scores)
            all_weights.append(weights)

        return (
            pd.concat(all_importances, ignore_index=True),
            pd.concat(all_scores, ignore_index=True),
            pd.concat(all_weights, ignore_index=True),
        )
