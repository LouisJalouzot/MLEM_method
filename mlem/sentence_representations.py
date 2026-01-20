from __future__ import annotations

import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    import torch

from exca import TaskInfra
from pydantic import ConfigDict, Field

from mlem.dataset import Dataset
from mlem.hidden_states import aggregate_masked_tensor, compute_hidden_states
from mlem.utils import BaseModel, get_device, seed_from_basemodel


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
    if token_aggregation == "none":
        mask = hidden_states.get_mask()
        if not mask.all():
            raise ValueError(
                "Token aggregation 'none' requires all sentences to have the same "
                "number of tokens (no padding). Found masked (padding) values."
            )
        # (n_sentences, max_seq_len, n_layers+1, hidden_size)
        data = hidden_states.get_data()
        n_stimuli = hidden_states.shape[0]
        n_layers = hidden_states.shape[2]
        # (n_sentences, n_layers+1, hidden_size)
        return data.swapaxes(1, 2).reshape(n_stimuli, n_layers, -1)
    else:
        # (n_sentences, n_layers+1, hidden_size)
        return aggregate_masked_tensor(hidden_states, dim=1, method=token_aggregation)


class SentenceRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    level: tp.Literal["sentence"] = "sentence"
    model_name: str = "bert-base-uncased"
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last", "none"] = (
        "mean"
    )
    add_special_tokens: bool = True

    layer: int = 5
    units: tp.List[int] | int | None = None
    noise_level: float = 0.0
    seed: int = 0

    device: tp.Optional[str] = None
    batch_size: int = 32
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    inner_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
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
            if isinstance(self.units, int):
                units = [self.units]
            else:
                units = self.units
            sentence_representations = sentence_representations[:, :, units]

        # (n_sentences, hidden_size)
        sentence_representations = sentence_representations[:, self.layer]

        scale = sentence_representations.std(dim=0)
        rng = np.random.default_rng(seed_from_basemodel(self))
        noise = rng.normal(scale=scale, size=sentence_representations.shape)
        noise *= self.noise_level

        return sentence_representations + torch.from_numpy(noise)

    @inner_infra.apply(exclude_from_cache_uid=["layer", "units", "noise_level", "seed"])
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

    # Dummy function to indicate successful computation without loading result in RAM
    @infra.apply(exclude_from_cache_uid=["layer", "units", "noise_level", "seed"])
    def precompute(self):
        self.forward()
