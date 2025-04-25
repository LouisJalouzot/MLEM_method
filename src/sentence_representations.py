import typing as tp

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
    sentences: tp.List[str]
    model_name: str = "bert-base-uncased"
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last"] = (
        "mean"
    )
    add_special_tokens: bool = True
    batch_size: int = 32
    device: tp.Optional[str] = None
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device",
        "batch_size",
    )

    def model_post_init(self, __context: tp.Any) -> None:
        if self.device is None:
            self.device = get_device()

    @infra.apply(exclude_from_cache_uid=["batch_size", "device"])
    def _compute_representations_cached(self) -> torch.Tensor:
        # (n_sentences, layers, hidden_size)
        return compute_sentence_representations(
            self.sentences,
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            add_special_tokens=self.add_special_tokens,
            token_aggregation=self.token_aggregation,
        )


class SentenceRepresentationsCfg(BaseModel):
    level: tp.Literal["sentence"] = "sentence"
    model_name: str = "bert-base-uncased"
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last"] = (
        "mean"
    )
    add_special_tokens: bool = True
    layer: int = 5
    units: tp.List[int] = None
    device: tp.Optional[str] = None
    batch_size: int = 32
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device",
        "batch_size",
    )

    def __call__(self, sentences: tp.List[str]) -> torch.Tensor:
        # (n_sentences, layers, hidden_size)
        sentence_representations = SentenceRepresentations(
            sentences=sentences,
            model_name=self.model_name,
            token_aggregation=self.token_aggregation,
            add_special_tokens=self.add_special_tokens,
            batch_size=self.batch_size,
            device=self.device,
            infra=self.infra,
        )._compute_representations_cached()

        if self.units is not None:
            sentence_representations = sentence_representations[
                :, :, self.units
            ]

        # (n_sentences, hidden_size)
        return sentence_representations[:, self.layer]
