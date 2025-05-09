import typing as tp

import torch
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from transformers import AutoConfig

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
    device: tp.Optional[str] = None
    batch_size: int = 32
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device",
        "batch_size",
    )

    def model_post_init(self, __context: tp.Any) -> None:
        if self.device is None:
            self.device = get_device()
        # config = AutoConfig.from_pretrained(self.model_name)
        # num_layers = (
        #     config.num_hidden_layers
        #     if hasattr(config, "num_hidden_layers")
        #     else config.num_layers
        # )
        # logger.debug(
        #     f"Model {self.model_name} has {num_layers} layers and {config.hidden_size} hidden size."
        # )
        # assert (
        #     self.layer <= num_layers
        # ), f"Layer {self.layer} is out of range for model {self.model_name} with {num_layers} layers."
        # if self.units is not None:
        #     assert (
        #         min(self.units) >= 0
        #     ), f"Units must be non-negative. Found {min(self.units)}."
        #     assert (
        #         max(self.units) < config.hidden_size
        #     ), f"Unit {max(self.units)} are out of range for model {self.model_name} with {config.hidden_size} hidden size."

    def __call__(self):
        # (n_sentences, n_layers+1, hidden_size)
        sentence_representations = self.forward()
        if self.units is not None:
            sentence_representations = sentence_representations[:, :, self.units]

        # (n_sentences, hidden_size)
        return sentence_representations[:, self.layer]

    @infra.apply(exclude_from_cache_uid=["layer", "units", "batch_size", "device"])
    def forward(self) -> torch.Tensor:
        # (n_sentences, n_layers+1, hidden_size)
        return compute_sentence_representations(
            sentences=self.dataset.sentences,
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            add_special_tokens=self.add_special_tokens,
            token_aggregation=self.token_aggregation,
        )
