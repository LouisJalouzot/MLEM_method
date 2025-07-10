from __future__ import annotations

import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

from exca import TaskInfra
from pydantic import ConfigDict

from src.utils import BaseModel, encode_df


class Dataset(BaseModel):
    path: str = "datasets/short_sentence.csv"
    seed: int = 0
    n_features_simu: int = 16
    n_samples_simu: int = 256
    _features: tp.List[str] = None
    _triu_indices: tp.Tuple[np.ndarray, np.ndarray] = None
    _pfeatures: tp.List[str] = None
    _level: str = None
    _df: pd.DataFrame = None
    _df_features: pd.DataFrame = None
    _sentences: tp.List[str] = None
    _words: tp.List[str] = None
    _sentence_id: tp.List[tp.Any] = None

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context):
        self.features

    def read(self, only_columns=False) -> pd.DataFrame:
        import pandas as pd

        if self.path == "simulated":
            features = np.array([f"Feat. {i+1}" for i in range(self.n_features_simu)])
            if only_columns:
                return features
            else:
                df = pd.DataFrame(
                    np.random.choice(
                        ["A", "B"], size=(self.n_samples_simu, len(features))
                    ),
                    columns=features,
                )

                return df
        elif self.path.endswith(".csv"):
            data = pd.read_csv(self.path, nrows=0 if only_columns else None)
            if only_columns:
                return data.columns.values
        else:
            if only_columns:
                from pyarrow import parquet

                return np.array([col.name for col in parquet.read_schema(self.path)])
            else:
                data = pd.read_parquet(self.path)
        return data

    @property
    def features(self) -> np.ndarray:
        if self._features is not None:
            return self._features
        else:
            features = self.read(only_columns=True)
            if "word" in features:
                features = features[
                    ~np.isin(features, ["word", "start_idx", "end_idx", "sentence"])
                ]
                self._level = "word"
            elif "sentence" in features:
                self._level = "sentence"
                features = features[features != "sentence"]
            elif "simulated" in self.path:
                self._level = "simulated"
            self._features = np.array(features, dtype=str)
            self._triu_indices = np.triu_indices(len(features))
            self._pfeatures = np.array(
                [
                    f"({features[i]} x {features[j]})" if i != j else features[i]
                    for i, j in zip(*self._triu_indices)
                ],
                dtype=str,
            )

            return self._features

    @property
    def triu_indices(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        self.features
        return self._triu_indices

    @property
    def pfeatures(self) -> tp.List[str]:
        self.features
        return self._pfeatures

    @property
    def df(self) -> pd.DataFrame:
        import pandas as pd

        if self._df is not None:
            return self._df
        else:
            df = self.read()
            # Add pairwise features
            pairwise_features = []
            for i, f_1 in enumerate(self.features):
                for f_2 in self.features[i + 1 :]:
                    s = df[f_1].astype(str) + ", " + df[f_2].astype(str)
                    s.name = f"({f_1} x {f_2})"
                    pairwise_features.append(s)
            df = pd.concat([df, *pairwise_features], axis=1)
            self._df = df

            return df

    @property
    def df_features(self) -> pd.DataFrame:
        return self.df[self.features]

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def level(self) -> str:
        self.features
        return self._level

    @property
    def sentences(self) -> tp.List[str]:
        return self.df.sentence.to_list()

    @property
    def words_df(self) -> pd.DataFrame:
        return self.df[["word", "sentence", "start_idx", "end_idx"]]

    @infra.apply
    def encode(self) -> torch.Tensor:
        return encode_df(self.df_features)
