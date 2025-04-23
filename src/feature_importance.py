import typing as tp
from time import time

import numpy as np
import pandas as pd
import torch

# Removed captum, statsmodels, tqdm imports
# from captum.attr import FeaturePermutation
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict

# from statsmodels.stats.descriptivestats import describe
# from tqdm.auto import tqdm

from src.trainer import Trainer
from src.utils import BaseModel

# Import core function and stats function
from src.core.feature_importance import (
    compute_feature_importance_core,
    compute_stats,
)

# Removed compute_stats function


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

        # Call the core function
        importances, spearman = compute_feature_importance_core(
            model=model,
            dataset=dataset,
            features=features,
            n_perm=self.n_perm,
            alpha=self.alpha,
            warn_ci=self.warn_ci,
        )

        return importances, spearman
