import typing as tp
from time import time

import pandas as pd
import torch
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Removed tqdm import
# from tqdm.auto import tqdm

from src.pairwise_dataset import PairwiseDatasetCfg
from src.sentence_representations import SentenceRepresentations
from src.spd_matrix_learner import SPDMatrixLearnerCfg
from src.stimulis import Stimulis
from src.utils import BaseModel
from src.word_representations import WordRepresentations

# Import core trainer function and core model/dataset types
from src.core.trainer import train_loop
from src.core.spd_matrix_learner import SPDMatrixLearner
from src.core.pairwise_dataset import PairwiseDataset


class Trainer(BaseModel):
    model: SPDMatrixLearnerCfg = SPDMatrixLearnerCfg()
    dataframe: Stimulis = Stimulis()
    representations: SentenceRepresentations | WordRepresentations = Field(
        default=SentenceRepresentations(), discriminator="level"
    )
    dataset: PairwiseDatasetCfg = PairwiseDatasetCfg()
    lr: float = 0.1
    weight_decay: float = 0
    max_epochs: int = 500
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    eps: float = 1e-5

    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _device: str | None = None

    @property
    def features(self) -> tp.List[str]:
        return self.dataframe._features

    def init(
        self, state_dict=None
    ) -> tp.Tuple[SPDMatrixLearner, PairwiseDataset]:  # Updated type hints
        torch.set_float32_matmul_precision("medium")
        if self._device is None:
            from src.utils import device

            self._device = device

        Y = self.representations.compute_representations(
            self.dataframe.stimulis
        )
        model = self.model.build(num_features=self.dataframe.num_features)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model = model.to(self._device)
        X = self.dataframe.encode().to(self._device)
        Y = Y.to(self._device)
        dataset = self.dataset.build(X, Y)

        return model, dataset

    @infra.apply
    def train(self) -> tp.Tuple[torch.Tensor, pd.DataFrame]:
        """
        Train a model with caching and optional remote execution.

        Args:
            model: The model to train
            dataloader: DataLoader providing batches

        Returns:
            Trained model state dict and logs
        """
        model, dataset = self.init()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            maximize=model.maximize,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=("max" if model.maximize else "min"),
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        # Call the core training loop
        model, logs = train_loop(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=self.max_epochs,
            eps=self.eps,
            device=self._device,
        )

        # Output state_dict as nn.Module can't be serialized for caching
        return model.state_dict(), logs
