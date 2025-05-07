import typing as tp
from pathlib import Path

import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict

from src.utils import BaseModel, encode_df


class Dataset(BaseModel):
    csv_path: str = "datasets/short_sentence.csv"
    _features: tp.List[str] = None
    _level: str = None
    _df: pd.DataFrame = None
    _df_features: pd.DataFrame = None
    _sentences: tp.List[str] = None
    _words: tp.List[str] = None
    _sentence_id: tp.List[tp.Any] = None
    infra: TaskInfra = TaskInfra(folder=".cache")

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context):
        self.features

    @property
    def df(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        else:
            self._df = pd.read_csv(self.csv_path)

            return self._df

    @property
    def df_features(self) -> pd.DataFrame:
        return self.df[self.features]

    @property
    def features(self) -> tp.List[str]:
        if self._features is not None:
            return self._features
        else:
            assert Path(self.csv_path).exists(), f"File {self.csv_path} does not exist"
            features = pd.read_csv(self.csv_path, nrows=0).columns.values
            if "word" in features:
                assert "sentence_id" in features
                features = features[features != "word"]
                features = features[features != "sentence_id"]
                self._level = "word"
                assert "sentence" not in features
            elif "sentence" in features:
                self._level = "sentence"
                features = features[features != "sentence"]
                assert "word" not in features and "sentence_id" not in features
            self._features = features

            return self._features

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def level(self) -> str:
        self.features
        return self._level

    @property
    def sentences(self) -> tp.List[str]:
        if self._sentences is not None:
            return self._sentences
        else:
            assert (
                self.level == "sentence"
            ), "sentences are only available for sentence level datasets"
            self._sentences = self.df.sentence.to_list()
            return self._sentences

    @property
    def words(self) -> tp.List[str]:
        if self._words is not None:
            return self._words
        else:
            assert (
                self.level == "word"
            ), "words are only available for word level datasets"
            self._words = self.df.word.to_list()
            return self._words

    @property
    def sentence_id(self) -> tp.List[tp.Any]:
        if self._sentence_id is not None:
            return self._sentence_id
        else:
            assert (
                self.level == "word"
            ), "sentence_id is only available for word level datasets"
            self._sentence_id = self.df.sentence_id.to_list()
            return self._sentence_id

    @infra.apply
    def encode(self) -> torch.Tensor:
        return encode_df(self.df_features)
