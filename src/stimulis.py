import typing as tp

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict

from src.utils import BaseModel, encode_df


class Stimulis(BaseModel):
    csv_path: str = "datasets/short_sentence.csv"
    _df: pd.DataFrame = None
    _features: tp.List[str] = None
    _sentences: tp.List[str] = None
    _words: tp.List[str] = None
    _sentence_id: tp.List[tp.Any] = None

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context):
        self._df = pd.read_csv(self.csv_path)
        if "word" in self._df.columns:
            self._words = self._df.word.tolist()
            self._sentence_id = self._df.sentence_id.tolist()
        elif "sentence" in self._df.columns:
            self._sentences = self._df.sentence.tolist()
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
    def sentences(self) -> tp.List[str]:
        return self._sentences

    @property
    def words(self) -> tp.List[str]:
        return self._words

    @property
    def sentence_id(self) -> tp.List[tp.Any]:
        return self._sentence_id

    def encode(self) -> torch.Tensor:
        return encode_df(self._df)
