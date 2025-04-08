import typing as tp

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict
from sklearn.preprocessing import MinMaxScaler

from src.utils import BaseModel


def encode_df(df: pd.DataFrame) -> torch.Tensor:
    X = np.zeros(df.shape, dtype=np.float32)
    number_cols = np.array([np.issubdtype(t, np.number) for t in df.dtypes])
    for i in range(df.shape[1]):
        s = df.iloc[:, i]
        if number_cols[i]:
            X[:, i] = s.values
        else:
            s = s.astype("category").cat.codes
            # -1 category code corresponds to NaN values
            s[s == -1] = np.nan
            X[:, i] = s
    X[:, number_cols] = MinMaxScaler().fit_transform(X[:, number_cols])

    return torch.from_numpy(X)


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
