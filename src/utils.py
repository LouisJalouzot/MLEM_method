import torch
import subprocess
import random
from pydantic import BaseModel
import typing as tp


def _get_free_gpu():
    gpus = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
    )
    gpus = gpus.decode("utf-8").split("\n")
    free_rams = tuple(map(lambda x: float(x.rstrip(" [MiB]")), gpus[1:-1]))
    max_free = max(free_rams)
    max_free_idxs = tuple(
        i for i in range(len(free_rams)) if abs(max_free - free_rams[i]) <= 200
    )

    return random.choice(max_free_idxs)


if torch.cuda.is_available():
    device = torch.device(f"cuda:{_get_free_gpu()}")
else:
    device = torch.device("cpu")


def nanmax(tensor, dim=None, keepdim=False):
    # From https://github.com/pytorch/pytorch/issues/61474#issuecomment-1735537507
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output


def nanmin(tensor, dim=None, keepdim=False):
    # From https://github.com/pytorch/pytorch/issues/61474#issuecomment-1735537507
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output


class BaseModel(BaseModel):
    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.model_dump() == other.model_dump()
