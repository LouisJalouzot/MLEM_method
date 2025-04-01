import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
import typing as tp


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
    df: str = "datasets/short_sentence.csv"
    _features: tp.List[str] = None
    stimulis: tp.List[tp.Any] = None

    def model_post_init(self, __context):
        self.df = pd.read_csv(self.df)
        if "word" in self.df.columns:
            self.stimulis = self.df["word"].values
        elif "sentence" in self.df.columns:
            self.stimulis = self.df["sentence"].values
        self.df = self.df.drop(columns=["word", "sentence"], errors="ignore")
        self._features = list(self.df.columns)

    @property
    def features(self) -> tp.List[str]:
        return self._features

    @property
    def num_features(self) -> int:
        return len(self.features)

    def encode(self):
        return encode_df(self.df)
