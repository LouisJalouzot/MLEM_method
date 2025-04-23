import typing as tp

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict

from src.utils import BaseModel
from src.core.stimulis import encode_df


class Stimulis(BaseModel):
    csv_path: str = "datasets/short_sentence.csv"
    _df: pd.DataFrame = None
    _features: tp.List[str] = None
    _stimulis: np.array = None

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context):
        self._df = pd.read_csv(self.csv_path)
        if "word" in self._df.columns:
            self._stimulis = self._df[["word", "sentence_id"]]
        elif "sentence" in self._df.columns:
            self._stimulis = self._df["sentence"].values
        self._df = self._df.drop(
            columns=["word", "sentence", "sentence_id"], errors="ignore"
        )
        self._features = self._df.columns.values

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def num_features(self) -> int:
        return len(self._features)

    @property
    def features(self) -> tp.List[str]:
        return self._features

    @property
    def stimulis(self) -> np.array:
        return self._stimulis

    def encode(self) -> torch.Tensor:
        return encode_df(self._df)
