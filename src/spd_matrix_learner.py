import typing as tp

import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import ConfigDict

from src.utils import BaseModel
from src.core.spd_matrix_learner import SPDMatrixLearner


class SPDMatrixLearnerCfg(BaseModel):
    param: str = "cholesky"
    fro_norm: bool = True
    init: tp.Optional[str] = None
    init_kwargs: dict = {}
    loss: str = "spearman"
    spearman_regularization: str = "l2"
    spearman_regularization_strength: float = 1.0

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def build(self, num_features) -> SPDMatrixLearner:
        """Build the model using this configuration"""
        return SPDMatrixLearner(
            num_features=num_features,
            param=self.param,
            fro_norm=self.fro_norm,
            init=self.init,
            init_kwargs=self.init_kwargs,
            loss=self.loss,
            spearman_regularization=self.spearman_regularization,
            spearman_regularization_strength=self.spearman_regularization_strength,
        )
