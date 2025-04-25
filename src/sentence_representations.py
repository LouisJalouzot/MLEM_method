import typing as tp
from typing import Any

import torch
from exca import TaskInfra
from pydantic import ConfigDict

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
    # (n_sentences, layers, max_seq_len, hidden_size)
    hidden_states = compute_hidden_states(
        sentences=sentences,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    # (n_sentences, layers, hidden_size)
    return aggregate_masked_tensor(
        hidden_states, dim=2, method=token_aggregation
    )


class SentenceRepresentations(BaseModel):
    """Computes sentence representations using a pre-trained transformer model.

    This class handles the computation of sentence embeddings by processing a list
    of sentences through a specified Hugging Face transformer model. It allows
    configuration of various parameters like model name, batch size, aggregation
    method for tokens, and selection of specific layers and units. Caching is
    used to avoid recomputing representations for the same configuration (excluding
    layer and unit selection).

    Attributes:
        sentences: A list of strings, where each string is a sentence.
        level: The level of representation, fixed to "sentence".
        model_name: The name of the Hugging Face model to use (e.g., "bert-base-uncased").
        token_aggregation: The method used to aggregate token hidden states into a
            sentence representation (e.g., "mean", "sum").
        add_special_tokens: Whether to include special tokens ([CLS], [SEP]) during
            tokenization.
        batch_size: The batch size used for processing sentences through the model.
        layer: The specific layer from which to extract the final sentence representations.
        units: An optional list of specific hidden unit indices to select from the
            chosen layer's representation. If None, all units are kept.
        infra: Configuration for caching mechanism.
        model_config: Pydantic model configuration.
    """

    sentences: tp.List[str]
    level: tp.Literal["sentence"] = "sentence"
    model_name: str = "bert-base-uncased"
    token_aggregation: str = "mean"
    add_special_tokens: bool = True
    batch_size: int = 32
    layer: int = 5
    units: tp.List[int] = None
    _device: tp.Optional[torch.device] = None
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        if self._device is None:
            self._device = get_device()

    @infra.apply(exclude_from_cache_uid=["layer", "units"])
    def _compute_representations_cached(self) -> torch.Tensor:
        # (n_sentences, layers, hidden_size)
        return compute_sentence_representations(
            self.sentences,
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self._device,
            add_special_tokens=self.add_special_tokens,
            token_aggregation=self.token_aggregation,
        )

    def __call__(self) -> torch.Tensor:
        # (n_sentences, layers, hidden_size)
        sentence_representations = self._compute_representations_cached()
        if self.units is not None:
            sentence_representations = sentence_representations[
                :, :, self.units
            ]

        # (n_sentences, hidden_size)
        return sentence_representations[:, self.layer]
