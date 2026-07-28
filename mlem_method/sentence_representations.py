from __future__ import annotations

import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    import torch

from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field

from .dataset import Dataset
from .hidden_states import (
    aggregate_masked_tensor,
    compute_hidden_states,
    subtract_word_mean_tokens,
)
from .utils import BaseModel, get_device, seed_from_basemodel


def compute_sentence_representations(
    sentences: tp.List[str],
    model_name: str = "prajjwal1/bert-tiny",
    batch_size: int = 32,
    device: str = "cpu",
    add_special_tokens: bool = True,
    token_aggregation: str = "mean",
    untrained: bool = False,
    revision: tp.Optional[str] = None,
) -> torch.Tensor:
    # (n_sentences, max_seq_len, n_layers+1, hidden_size)
    hidden_states = compute_hidden_states(
        sentences=sentences,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        add_special_tokens=add_special_tokens,
        untrained=untrained,
        revision=revision,
        token_aggregation=None if token_aggregation == "none" else token_aggregation,
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
    # (n_sentences, n_layers+1, hidden_size)
    return hidden_states


class SentenceRepresentations(BaseModel):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    level: tp.Literal["sentence"] = "sentence"
    model_name: str = "bert-base-uncased"
    revision: tp.Optional[str] = None
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last", "none"] = (
        "mean"
    )
    add_special_tokens: bool = True
    untrained: bool = False
    normalize_embeddings: tp.Literal["none", "layer0", "diff"] = "none"
    normalize_by_word: bool = False

    layer: int | tp.List[int] | None = 5
    units: int | tp.List[int] | None = None
    pca: int | float | None = None
    svd_solver: str = "randomized"
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

    def model_post_init(self, context):
        super().model_post_init(context)
        assert not isinstance(self.pca, float) or self.svd_solver == "full"

    def __call__(self):
        return self._get_final_representations()

    def _get_final_representations(self) -> torch.Tensor:
        import torch

        # (n_sentences, n_layers+1, hidden_size)
        sentence_representations = self.forward()

        # Apply embedding normalization
        if self.normalize_embeddings == "layer0":
            # Subtract embedding layer (layer 0) from all layers
            sentence_representations -= sentence_representations[:, 0:1]
        elif self.normalize_embeddings == "diff":
            # Subtract previous layer from each layer
            sentence_representations[:, 1:] -= sentence_representations[:, :-1].clone()
        if self.units is not None:
            if isinstance(self.units, int):
                units = [self.units]
            else:
                units = self.units
            sentence_representations = sentence_representations[:, :, units]

        if self.layer is not None:
            sentence_representations = sentence_representations[:, self.layer]
        else:
            # Remove embedding layer
            sentence_representations = sentence_representations[:, 1:]
        # (n_sentences, hidden_size)
        n_stimuli = sentence_representations.shape[0]
        sentence_representations = sentence_representations.reshape(n_stimuli, -1)

        # Apply PCA if requested (int for n_components, float for variance ratio)
        if self.pca is not None:
            from sklearn.decomposition import PCA

            original_shape = sentence_representations.shape
            pca_model = PCA(
                n_components=self.pca,
                random_state=self.seed,
                svd_solver=self.svd_solver,
            )
            sentence_representations = pca_model.fit_transform(sentence_representations)
            if isinstance(self.pca, float):
                logger.info(
                    f"Applied {self.pca * 100:.1f}% PCA: reduced from {original_shape[1]} to {self.pca} dimensions"
                )
            else:
                logger.info(
                    f"Applied {self.pca} components PCA: reduced from {original_shape[1]} and retained {pca_model.explained_variance_ratio_.sum() * 100:.1f}% of variance"
                )
            sentence_representations = torch.from_numpy(sentence_representations)

        scale = sentence_representations.std(dim=0)
        rng = np.random.default_rng(seed_from_basemodel(self))
        noise = rng.normal(scale=scale, size=sentence_representations.shape)
        noise *= self.noise_level

        return sentence_representations + torch.from_numpy(noise)

    @inner_infra.apply(
        exclude_from_cache_uid=[
            "layer",
            "units",
            "noise_level",
            "seed",
            "normalize_embeddings",
            "normalize_by_word",
        ]
    )
    def forward(self) -> torch.Tensor:
        import torch
        from torch.masked import masked_tensor

        # (n_sentences, n_layers+1, hidden_size)
        if self.normalize_by_word:
            # Get token-level data to normalize per position and token type
            hidden_states_masked, input_ids = compute_hidden_states(
                sentences=self.dataset.sentences,
                model_name=self.model_name,
                batch_size=self.batch_size,
                device=self.device or get_device(),
                add_special_tokens=self.add_special_tokens,
                return_input_ids=True,
                untrained=self.untrained,
                revision=self.revision,
            )
            # (n_sentences, max_seq_len) bool
            attention_mask = hidden_states_masked.get_mask()[:, :, 0, 0]
            # (n_sentences, max_seq_len, n_layers+1, hidden_size) — mutable copy
            data = hidden_states_masked.get_data().clone()

            subtract_word_mean_tokens(data, input_ids, attention_mask)

            # Re-mask and aggregate
            attn_mask_4d = attention_mask[:, :, None, None].broadcast_to(data.shape)
            hidden_states_masked = masked_tensor(data, attn_mask_4d)

            if self.token_aggregation == "none":
                if not attn_mask_4d.all():
                    raise ValueError(
                        "Token aggregation 'none' requires all sentences to have the same "
                        "number of tokens (no padding). Found masked (padding) values."
                    )
                n_stimuli = data.shape[0]
                n_layers = data.shape[2]
                return data.swapaxes(1, 2).reshape(n_stimuli, n_layers, -1)
            else:
                return aggregate_masked_tensor(
                    hidden_states_masked, dim=1, method=self.token_aggregation
                )
        else:
            return compute_sentence_representations(
                sentences=self.dataset.sentences,
                model_name=self.model_name,
                batch_size=self.batch_size,
                device=self.device or get_device(),
                add_special_tokens=self.add_special_tokens,
                token_aggregation=self.token_aggregation,
                untrained=self.untrained,
                revision=self.revision,
            )

    # Dummy function to indicate successful computation without loading result in RAM
    @infra.apply(
        exclude_from_cache_uid=[
            "layer",
            "units",
            "noise_level",
            "seed",
            "normalize_embeddings",
            "normalize_by_word",
        ]
    )
    def precompute(self):
        self.forward()
