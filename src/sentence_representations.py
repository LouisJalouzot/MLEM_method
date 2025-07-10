from __future__ import annotations

import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    import torch

from exca import TaskInfra
from pydantic import ConfigDict, Field

from src.dataset import Dataset
from src.hidden_states import aggregate_masked_tensor, compute_hidden_states
from src.utils import BaseModel, get_device


def compute_sentence_representations(
    sentences: tp.List[str],
    model_name: str = "prajjwal1/bert-tiny",
    batch_size: int = 32,
    device: str = "cpu",
    add_special_tokens: bool = True,
    token_aggregation: str = "mean",
) -> torch.Tensor:
    # (n_sentences, max_seq_len, n_layers+1, hidden_size)
    hidden_states = compute_hidden_states(
        sentences=sentences,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    # (n_sentences, n_layers+1, hidden_size)
    return aggregate_masked_tensor(hidden_states, dim=1, method=token_aggregation)


class SentenceRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    level: tp.Literal["sentence"] = "sentence"
    model_name: str = "bert-base-uncased"
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last"] = "mean"
    add_special_tokens: bool = True

    layer: int = 5
    units: tp.List[int] = None
    noise_level: float = 0.0
    seed: int = 0

    device: tp.Optional[str] = None
    batch_size: int = 32
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "batch_size",
        "device",
    )

    def __call__(self):
        import torch

        # (n_sentences, n_layers+1, hidden_size)
        sentence_representations = self.forward()
        if self.units is not None:
            sentence_representations = sentence_representations[:, :, self.units]

        # (n_sentences, hidden_size)
        sentence_representations = sentence_representations[:, self.layer]

        noise = np.random.normal(size=sentence_representations.shape) * self.noise_level

        return sentence_representations + torch.from_numpy(noise)

    @infra.apply(exclude_from_cache_uid=["layer", "units", "noise_level", "seed"])
    def forward(self) -> torch.Tensor:
        # (n_sentences, n_layers+1, hidden_size)
        return compute_sentence_representations(
            sentences=self.dataset.sentences,
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self.device or get_device(),
            add_special_tokens=self.add_special_tokens,
            token_aggregation=self.token_aggregation,
        )
