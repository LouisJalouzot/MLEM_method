import logging
import random
import subprocess
import typing as tp

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pydantic import BaseModel
from pydantic import BaseModel as _BaseModel
from pydantic import ValidationError, model_validator
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
            i for i in range(len(free_rams)) if abs(max_free - free_rams[i]) <= 200
        )
        gpu_id = random.choice(max_free_idxs)

        return f"cuda:{gpu_id}"
    else:
        return "cpu"


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

    # Configuration mapping: {shared_field_name: [dependent_field_name_1, ...]}
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {}

    @model_validator(mode="before")
    @classmethod
    def _inject_shared_instances(cls, data: tp.Any) -> tp.Any:
        """
        Handles instantiation or adoption of shared fields and injection into dependents.
        Modifies the input data dictionary directly.
        Runs before standard Pydantic validation.
        """
        # Only process dict data and if config is defined
        if not isinstance(data, dict) or not cls._shared_fields_config:
            return data

        shared_instances: tp.Dict[str, BaseModel] = {}

        # 1. Instantiate or adopt shared fields from top-level data
        for shared_field_name in cls._shared_fields_config.keys():
            if shared_field_name not in cls.model_fields:
                raise ValueError(
                    f"Config Error: Shared field '{shared_field_name}' not in {cls.__name__}"
                )

            shared_field_type = cls.model_fields[shared_field_name].annotation
            # Ensure the shared field is a Pydantic model itself
            if not (
                isinstance(shared_field_type, type)
                and issubclass(shared_field_type, BaseModel)
            ):
                logger.warning(
                    f"Skipping injection: Shared field '{shared_field_name}' is not a BaseModel subclass."
                )
                continue

            shared_field_data_or_instance = data.pop(
                shared_field_name, {}
            )  # Default to {} if missing

            instance = None
            # Check if an instance of the correct type was provided
            if isinstance(shared_field_data_or_instance, shared_field_type):
                instance = shared_field_data_or_instance
            # Check if a dictionary was provided for instantiation
            elif isinstance(shared_field_data_or_instance, dict):
                try:
                    # Create the shared instance from dict
                    instance = shared_field_type(**shared_field_data_or_instance)
                except ValidationError as e:
                    raise ValueError(
                        f"Validation failed for shared field '{shared_field_name}' from dict"
                    ) from e
            # Handle unexpected data type
            else:
                logger.warning(
                    f"Data for shared field '{shared_field_name}' is neither a valid instance nor a dict. Using defaults."
                )
                # Attempt to create with defaults
                try:
                    instance = shared_field_type()
                except ValidationError as e:
                    # If even default instantiation fails, raise error
                    raise ValueError(
                        f"Default instantiation failed for shared field '{shared_field_name}'"
                    ) from e

            if instance:
                shared_instances[shared_field_name] = instance
                # Place the instance (either adopted or created) back into data for Pydantic
                data[shared_field_name] = instance

        # 2. Inject shared instances into dependent fields' initialization data
        for (
            shared_field_name,
            dependent_field_names,
        ) in cls._shared_fields_config.items():
            if shared_field_name not in shared_instances:
                continue  # Skip if shared instance creation/adoption failed

            shared_instance = shared_instances[shared_field_name]

            for dependent_field_name in dependent_field_names:
                if dependent_field_name not in cls.model_fields:
                    logger.warning(
                        f"Config Warning: Dependent field '{dependent_field_name}' not found in {cls.__name__}. Skipping."
                    )
                    continue

                # Get data intended for the dependent field, default to {}
                dependent_data = data.get(dependent_field_name, {})

                # Check if dependent data is already an instance - raise error
                if isinstance(dependent_data, BaseModel):
                    raise ValueError(
                        f"Cannot inject shared field '{shared_field_name}'. "
                        f"Data for dependent field '{dependent_field_name}' is already an instance."
                    )
                # Expect a dict here to inject the shared instance for initialization
                elif not isinstance(dependent_data, dict):
                    logger.warning(
                        f"Data for dependent field '{dependent_field_name}' is not a dict. Using defaults for injection."
                    )
                    dependent_data = {}

                # Warn if the shared field key exists redundantly in the dependent data
                if shared_field_name in dependent_data:
                    logger.warning(
                        f"Shared field '{shared_field_name}' defined redundantly in '{dependent_field_name}' data. "
                        f"Top-level definition/instance will be used."
                    )

                # Prepare the data dict for the dependent field, injecting the shared instance
                dependent_data_with_injection = dependent_data.copy()
                # Inject the shared instance object itself
                dependent_data_with_injection[shared_field_name] = shared_instance
                data[dependent_field_name] = dependent_data_with_injection

        # Return the modified data for Pydantic to perform final validation and instantiation
        return data
