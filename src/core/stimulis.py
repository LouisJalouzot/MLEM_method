import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


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
