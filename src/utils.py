import random
import subprocess
import sys
import typing as tp

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pydantic import BaseModel as _BaseModel
from pydantic import model_validator
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.descriptivestats import describe

logger.remove()
logger.add(sink=sys.stdout, level="DEBUG")

np.random.seed(0)
torch.manual_seed(0)

DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    if torch.cuda.is_available():
        # Get GPU id with the most free memory, select at random if there are multiple
        try:
            gpus = subprocess.check_output(
                ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
            )
            gpus = gpus.decode("utf-8").split("\n")
            free_rams = tuple(map(lambda x: float(x.rstrip(" [MiB]")), gpus[1:-1]))
            max_free = max(free_rams)
            max_free_idxs = tuple(
                i for i in range(len(free_rams)) if abs(max_free - free_rams[i]) <= 200
            )
            gpu_id = random.choice(max_free_idxs)
            DEVICE = f"cuda:{gpu_id}"
        except:
            DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    return DEVICE


def encode_df(df: pd.DataFrame) -> torch.Tensor:
    if df.empty:
        # Return an empty tensor with the correct number of columns but 0 rows
        return torch.empty((0, df.shape[1]), dtype=torch.float32)

    X = np.zeros(df.shape, dtype=np.float32)
    number_cols = np.array(
        [
            (not pd.api.types.is_categorical_dtype(t) and np.issubdtype(t, np.number))
            for t in df.dtypes
        ]
    )
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


class BaseModel(_BaseModel):
    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.model_dump() == other.model_dump()


class BaseModelSharing(BaseModel):
    """
    Base class enabling automatic injection of shared field instances.

    To use:
    1. Inherit from this class.
    2. Define `_shared_fields_config`: ClassVar mapping shared field names
       to lists of dependent field names that require the shared instance.
    3. Define the actual fields corresponding to the names used in the config.
    """

    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {}

    @model_validator(mode="before")
    @classmethod
    def _inject_shared_instances(cls, data: tp.Any) -> tp.Any:
        """
        Handles instantiation/adoption of shared fields and injects them into dependents.
        Modifies the input data dictionary directly. Runs before Pydantic validation.
        """
        if not isinstance(data, dict) or not cls._shared_fields_config:
            return data

        for shared_field_name, dependent_field_names in cls._shared_fields_config.items():
            # 1. Validate and Get/Create the Shared Instance
            if shared_field_name not in cls.model_fields:
                raise ValueError(
                    f"Config Error: Shared field '{shared_field_name}' in "
                    f"_shared_fields_config not found as a field in {cls.__name__}."
                )

            shared_field_type = cls.model_fields[shared_field_name].annotation
            if not (
                isinstance(shared_field_type, type)
                and issubclass(shared_field_type, _BaseModel)
            ):
                raise TypeError(
                    f"Config Error: Shared field '{shared_field_name}' in {cls.__name__} "
                    f"must be a Pydantic BaseModel subclass, got {shared_field_type}."
                )

            # Retrieve data for the shared field, defaulting to {} for default instantiation
            shared_data = data.get(shared_field_name, {})

            if not (
                isinstance(shared_data, shared_field_type)
                or isinstance(shared_data, dict)
            ):
                raise TypeError(
                    f"Invalid data for shared field '{shared_field_name}'. "
                    f"Expected a {shared_field_type.__name__} instance or a dict, "
                    f"got {type(shared_data).__name__}."
                )

            # Place the resolved shared instance into data for Pydantic validation
            data[shared_field_name] = shared_data

            # 2. Inject the Shared Instance into Dependent Fields' Initialization Data
            for dependent_field_name in dependent_field_names:
                if dependent_field_name not in cls.model_fields:
                    raise ValueError(
                        f"Config Error: Dependent field '{dependent_field_name}' in "
                        f"_shared_fields_config not found as a field in {cls.__name__}."
                    )

                dependent_data = data.get(dependent_field_name, {})

                if isinstance(dependent_data, _BaseModel):
                    raise ValueError(
                        f"Injection Error: Cannot inject shared field '{shared_field_name}'. "
                        f"Data for dependent field '{dependent_field_name}' is already an instance "
                        f"of {type(dependent_data).__name__}, not a dict for initialization."
                    )

                if not isinstance(dependent_data, dict):
                    raise TypeError(
                        f"Injection Error: Data for dependent field '{dependent_field_name}' "
                        f"must be a dict for initialization, got {type(dependent_data).__name__}."
                    )

                # Inject the shared instance into the dependent's data
                dependent_data[shared_field_name] = shared_data
                data[dependent_field_name] = dependent_data

        return data


def compute_stats(data, alpha=0.01):
    """Compute descriptive statistics with confidence intervals"""
    return describe(
        data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha
    ).T
