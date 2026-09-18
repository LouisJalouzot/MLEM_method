from __future__ import annotations

import typing as tp
from functools import cached_property

import numpy as np

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

from exca import TaskInfra
from pydantic import ConfigDict

from .simulation import Simulation
from .utils import BaseModel, encode_df


def _pair_names(names: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(len(names))
    return np.array(
        [f"({names[i]} x {names[j]})" if names[i] != names[j] else names[i] for i, j in zip(*indices)],
        dtype=str,
    )


class Dataset(BaseModel):
    path: str = "datasets/short_sentence.csv"
    seed: int = 0
    mahalanobis: bool = False
    simulation: Simulation | None = None
    _level: str = None

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry", version="3")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def _exclude_from_cls_uid(self) -> tuple[str, ...]:
        return ("path",) if self.simulation is not None else ()

    def model_post_init(self, __context, /):
        _ = self.features

    def read(self, only_columns=False) -> pd.DataFrame:
        import pandas as pd

        if self.simulation is not None:
            if only_columns:
                return self.simulation.feature_names()
            return self.simulation.make_df(self.seed)
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

    @cached_property
    def features(self) -> np.ndarray:
        features = self.read(only_columns=True)
        if "word" in features:
            features = features[~np.isin(features, ["word", "start_idx", "end_idx", "sentence"])]
            self._level = "word"
        elif "sentence" in features:
            self._level = "sentence"
            features = features[features != "sentence"]
        elif self.simulation is not None:
            self._level = "simulated"
        return np.array(features, dtype=str)

    @cached_property
    def triu_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return np.triu_indices(len(self.features))

    @cached_property
    def pfeatures(self) -> list[str]:
        return _pair_names(self.features)

    @cached_property
    def df(self) -> pd.DataFrame:
        import pandas as pd

        df = self.read()
        # Add pairwise features
        pairwise_features = []
        for i, f_1 in enumerate(self.features):
            for f_2 in self.features[i + 1 :]:
                s = df[f_1].astype(str) + ", " + df[f_2].astype(str)
                s.name = f"({f_1} x {f_2})"
                pairwise_features.append(s)
        return pd.concat([df, *pairwise_features], axis=1)

    @property
    def df_features(self) -> pd.DataFrame:
        return self.df[self.features]

    @property
    def n_features(self) -> int:
        return len(self.features)

    @cached_property
    def _coordinate_groups(self) -> pd.Series:
        return self.encode()[1]

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinate_groups.index.to_numpy(dtype=str)

    @property
    def coordinate_groups(self) -> np.ndarray:
        return self._coordinate_groups.to_numpy(dtype=str)

    @property
    def n_coordinates(self) -> int:
        return len(self.coordinates)

    @cached_property
    def pcoordinates(self) -> np.ndarray:
        return _pair_names(self.coordinates)

    @property
    def pcoordinate_groups(self) -> np.ndarray:
        return _pair_names(self.coordinate_groups)

    @property
    def level(self) -> str:
        _ = self.features
        return self._level

    @property
    def sentences(self) -> list[str]:
        return self.df.sentence.to_list()

    @property
    def words_df(self) -> pd.DataFrame:
        return self.df[["word", "sentence", "start_idx", "end_idx"]]

    def encode(self) -> tuple[torch.Tensor, pd.Series]:
        return self._encode()

    @infra.apply
    def _encode(self) -> tuple[torch.Tensor, pd.Series]:
        return encode_df(self.df_features, simplex=self.mahalanobis)


class SimulatedRepresentations(BaseModel):
    dataset: Dataset
    level: tp.Literal["simulated"] = "simulated"
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @property
    def W(self):
        return None if self.dataset.simulation is None else getattr(self.dataset.simulation, "W", None)

    @property
    def gt_weights(self):
        return None if self.dataset.simulation is None else getattr(self.dataset.simulation, "gt_weights", None)

    def __call__(self):
        if self.dataset.simulation is None:
            raise ValueError("SimulatedRepresentations requires dataset.simulation")
        Z, groups = self.dataset.encode()
        return self.dataset.simulation.make_Y(Z, groups, self.dataset.seed, signed=self.dataset.mahalanobis)
