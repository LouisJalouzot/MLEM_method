import random
import subprocess
import typing as tp

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler


def get_device():
    if torch.cuda.is_available():
        # Get GPU id with the most free memory, select at random if there are multiple
        gpus = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
        )
        gpus = gpus.decode("utf-8").split("\n")
        free_rams = tuple(map(lambda x: float(x.rstrip(" [MiB]")), gpus[1:-1]))
        max_free = max(free_rams)
        max_free_idxs = tuple(
            i
            for i in range(len(free_rams))
            if abs(max_free - free_rams[i]) <= 200
        )
        gpu_id = random.choice(max_free_idxs)

        return f"cuda:{gpu_id}"
    else:
        return "cpu"


class BaseModel(BaseModel):
    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.model_dump() == other.model_dump()


def encode_df(df: pd.DataFrame) -> torch.Tensor:
    if df.empty:
        # Return an empty tensor with the correct number of columns but 0 rows
        return torch.empty((0, df.shape[1]), dtype=torch.float32)

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
    # Only apply MinMaxScaler if there are numeric columns
    if np.any(number_cols):
        X[:, number_cols] = MinMaxScaler().fit_transform(X[:, number_cols])

    return torch.from_numpy(X)
