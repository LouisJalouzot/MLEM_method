from __future__ import annotations

import typing as tp

from pydantic import ConfigDict

from .utils import BaseModel

if tp.TYPE_CHECKING:
    from .spd_matrix_learner_torch import SPDMatrixLearner


class SPDMatrixLearnerBuilder(BaseModel):
    param: tp.Literal["none", "diagonal", "sym", "triu", "exp", "cholesky", "dnn"] = "cholesky"
    fro_norm: bool = True
    loss: tp.Literal["spearman", "mse"] = "spearman"
    scoring: tp.Literal["spearman", "mse"] = "spearman"
    spearman_regularization: str = "l2"
    spearman_regularization_strength: float = 1.0

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def build(self, n_features) -> SPDMatrixLearner:
        """Build the model using this configuration"""
        from .spd_matrix_learner_torch import SPDMatrixLearner

        return SPDMatrixLearner(
            n_features=n_features,
            param=self.param,
            fro_norm=self.fro_norm,
            loss=self.loss,
            scoring=self.scoring,
            spearman_regularization=self.spearman_regularization,
            spearman_regularization_strength=self.spearman_regularization_strength,
        )
