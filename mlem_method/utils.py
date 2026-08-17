from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import typing as tp

# Set environment variable for deterministic CuBLAS operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch.multiprocessing as mp
from loguru import logger
from pydantic import BaseModel as _BaseModel
from pydantic import model_validator

# Use spawn for multiprocessing to avoid CUDA re-initialization error
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

logger.remove()
level = os.getenv("LOGGER_LEVEL", "INFO").upper()
logger.add(sink=sys.stdout, level=level)


def get_device():
    import torch

    if torch.cuda.is_available():
        # Get GPU id with the most free memory, select at random if there are multiple
        try:
            gpus = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
            gpus = gpus.decode("utf-8").split("\n")
            free_rams = tuple(float(x.rstrip(" [MiB]")) for x in gpus[1:-1])
            max_free = max(free_rams)
            max_free_idxs = tuple(i for i in range(len(free_rams)) if abs(max_free - free_rams[i]) <= 200)
            gpu_id = random.choice(max_free_idxs)
            return f"cuda:{gpu_id}"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return "cuda"
    else:
        return "cpu"


def _get_layers_from_config(config) -> int | None:
    """Try to extract layer count from a config object."""
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    return None


def get_n_layers(model_name: str, revision: str | None = None) -> int:
    """Get the number of hidden layers from a HuggingFace model config.

    Handles different config attribute names across model architectures:
    - num_hidden_layers (BERT, DeBERTa, Mamba, etc.)
    - n_layer (GPT-2, Bloom, etc.)
    - num_layers (T5, etc.)

    Also handles encoder-decoder models (e.g., T5Gemma, BART) and multimodal
    models with text_config (e.g., VLMs, Ministral).

    Args:
        model_name: HuggingFace model identifier.
        revision: The specific model revision or checkpoint to use. Defaults to None (uses main).

    Returns:
        Number of hidden layers in the model.

    Raises:
        ValueError: If no layer count attribute is found in the config.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision=revision)

    # Try top-level config
    if (layers := _get_layers_from_config(config)) is not None:
        return layers

    # For encoder-decoder or multimodal models, check sub-configs
    sub_configs = (
        getattr(config, "encoder", None),
        getattr(config, "decoder", None),
        getattr(config, "text_config", None),
    )
    for sub_config in sub_configs:
        if sub_config is not None and (layers := _get_layers_from_config(sub_config)) is not None:
            return layers

    raise ValueError(
        f"Could not determine number of layers for model '{model_name}'. "
        f"Config has no 'num_hidden_layers', 'n_layer', or 'num_layers' attribute "
        f"at top level or in encoder/decoder/text_config sub-configs."
    )


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_df(df: pd.DataFrame, simplex: bool = False) -> tuple[torch.Tensor, pd.Series]:
    import pandas as pd
    import torch
    from sklearn.preprocessing import MinMaxScaler

    coordinates = [str(column) for column in df.columns]
    if df.empty and not simplex:
        X = torch.empty((0, df.shape[1]), dtype=torch.float32)
        return X, pd.Series(coordinates, index=coordinates, name="Group", dtype=str)

    number_cols = np.array(
        [
            (
                not pd.api.types.is_categorical_dtype(t)
                and not pd.api.types.is_string_dtype(t)
                and np.issubdtype(t, np.number)
            )
            for t in df.dtypes
        ]
    )
    if simplex:
        from scipy.linalg import helmert

        columns = []
        coordinates = []
        groups = []
        for is_number, (feature, s) in zip(number_cols, df.items()):
            feature = str(feature)
            if is_number:
                z = s.to_numpy(dtype=np.float32)[:, None]
                if len(s):
                    z = MinMaxScaler().fit_transform(z)
                names = [feature]
            else:
                categorical = s.astype("category")
                codes = categorical.cat.codes.to_numpy()
                n_levels = len(categorical.cat.categories)
                if n_levels > 1:
                    z = (helmert(n_levels).T[codes] / np.sqrt(2)).astype(np.float32)
                    z[codes < 0] = np.nan
                else:
                    z = np.empty((len(s), 0), dtype=np.float32)
                names = [f"{feature}_{i}" for i in range(z.shape[1])]
            columns.append(z)
            coordinates.extend(names)
            groups.extend([feature] * len(names))

        X = np.hstack(columns) if columns else np.empty((len(df), 0), dtype=np.float32)
        return torch.from_numpy(X), pd.Series(groups, index=coordinates, name="Group", dtype=str)

    X = np.zeros(df.shape, dtype=np.float32)
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

    return torch.from_numpy(X), pd.Series(coordinates, index=coordinates, name="Group", dtype=str)


class BaseModel(_BaseModel):
    def __eq__(self, other: object) -> bool:
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

    _shared_fields_config: tp.ClassVar[dict[str, list[str]]] = {}

    @model_validator(mode="before")
    @classmethod
    def _inject_shared_instances(cls, data: tp.Any) -> tp.Any:
        """
        Handles instantiation/adoption of shared fields and injects them into dependents.
        Modifies the input data dictionary directly. Runs before Pydantic validation.
        """
        if not isinstance(data, dict) or not cls._shared_fields_config:
            return data

        for (
            shared_field_name,
            dependent_field_names,
        ) in cls._shared_fields_config.items():
            # 1. Validate and Get/Create the Shared Instance
            if shared_field_name not in cls.model_fields:
                raise ValueError(
                    f"Config Error: Shared field '{shared_field_name}' in "
                    f"_shared_fields_config not found as a field in {cls.__name__}."
                )

            shared_field_type = cls.model_fields[shared_field_name].annotation
            if not (isinstance(shared_field_type, type) and issubclass(shared_field_type, _BaseModel)):
                raise TypeError(
                    f"Config Error: Shared field '{shared_field_name}' in {cls.__name__} "
                    f"must be a Pydantic BaseModel subclass, got {shared_field_type}."
                )

            # Retrieve data for the shared field, defaulting to {} for default instantiation
            shared_data = data.get(shared_field_name, {})

            if not isinstance(shared_data, (shared_field_type, dict)):
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

                dependent_field = cls.model_fields[dependent_field_name]
                dependent_data = data.get(dependent_field_name, {})

                if isinstance(dependent_data, _BaseModel):
                    raise TypeError(
                        f"Injection Error: Cannot inject shared field '{shared_field_name}'. "
                        f"Data for dependent field '{dependent_field_name}' is already an instance "
                        f"of {type(dependent_data).__name__}, not a dict for initialization."
                    )

                if not isinstance(dependent_data, dict):
                    raise TypeError(
                        f"Injection Error: Data for dependent field '{dependent_field_name}' "
                        f"must be a dict for initialization, got {type(dependent_data).__name__}."
                    )

                # Skip injection for discriminated union fields with empty data,
                # allowing them to use their default factories with proper discriminators.
                # For regular fields, always inject even when data is empty.
                is_discriminated_union = getattr(dependent_field, "discriminator", None) is not None
                if is_discriminated_union and not dependent_data:
                    continue

                dependent_data[shared_field_name] = shared_data
                data[dependent_field_name] = dependent_data

        return data


def compute_stats(data, alpha=0.01):
    """Compute descriptive statistics with confidence intervals"""
    from statsmodels.stats.descriptivestats import describe

    return describe(data, stats=["mean", "std", "std_err", "ci"], use_t=True, alpha=alpha).T


def seed_from_basemodel(model: BaseModel):
    def strip_mahalanobis(value):
        if isinstance(value, dict):
            return {key: strip_mahalanobis(item) for key, item in value.items() if key != "mahalanobis"}
        if isinstance(value, (list, tuple)):
            return [strip_mahalanobis(item) for item in value]
        return value

    config_dict = strip_mahalanobis(model.model_dump())
    config_dict.pop("infra", None)
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    return int(config_hash, 16) % (2**32)


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient between two 1D tensors"""
    x_n = x - x.mean()
    y_n = y - y.mean()
    x_n = x_n / x_n.norm()
    y_n = y_n / y_n.norm()

    return (x_n * y_n).sum()


def spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spearman correlation coefficient between two 1D tensors"""
    dtype = x.dtype
    x_rank = x.argsort().argsort().to(dtype)
    y_rank = y.argsort().argsort().to(dtype)

    return corrcoef(x_rank, y_rank)
