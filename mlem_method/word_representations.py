from __future__ import annotations

import string
import typing as tp

if tp.TYPE_CHECKING:
    import pandas as pd
    import torch

import numpy as np
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field

from .dataset import Dataset
from .hidden_states import aggregate_masked_tensor, compute_hidden_states
from .utils import BaseModel, get_device, seed_from_basemodel


def cum_join_index(words):
    s = ""
    splits_idx = [0]
    for i, w in enumerate(words):
        if w not in string.punctuation and i > 0:
            s += " "
        s += w
        splits_idx.append(len(s))

    return s, splits_idx[:-1], splits_idx[1:]


def compute_word_representations(
    words: pd.DataFrame,
    model_name: str = "prajjwal1/bert-tiny",
    token_aggregation: str = "mean",
    batch_size: int = 32,
    device: str = "cpu",
    add_special_tokens: bool = True,
    untrained: bool = False,
    revision: tp.Optional[str] = None,
) -> torch.Tensor:
    """Computes word representations from hidden states by aggregating token representations.

    Args:
        words (pd.DataFrame): DataFrame containing columns:
            - 'word' (str): the token text
            - 'sentence' (str): full context sentence for this token
            - 'start_idx' (int): start character index of the token in sentence
            - 'end_idx' (int): end character index of the token in sentence
          Each row is one word occurrence; no grouping by sentence_id is needed.
        model_name (str, optional): Name of the Hugging Face model to use.
            Defaults to "prajjwal1/bert-tiny".
        token_aggregation (str, optional): Method to aggregate token representations
            ('mean', 'sum', etc.). Defaults to "mean".
        batch_size (int, optional): Batch size for processing sentences. Defaults to 32.
        device (str, optional): Device to run computations on ('cpu', 'cuda').
            Defaults to "cpu".
        add_special_tokens (bool, optional): Whether to include special tokens
            ([CLS], [SEP]) during tokenization. Defaults to True.
        untrained (bool, optional): If True, use a randomly initialized model.
            Defaults to False.
        revision (str, optional): The specific model revision or checkpoint to use.
            Defaults to None (uses main).

    Returns:
        torch.Tensor: A tensor containing word representations for all words,
        shaped (n_words, n_layers+1, hidden_size).

    Example:
        ```python
        import pandas as pd
        words_df = pd.DataFrame({
            'word': ['example', 'sentence'],
            'sentence': [
                'This is an example sentence.',
                'Another example sentence!'
            ],
            'start_idx': [10, 15],
            'end_idx': [18, 24]
        })
        reps = compute_word_representations(words_df)
        ```
    """
    import torch
    from torch.masked import masked_tensor

    sentences = words.sentence.tolist()
    # (n_words)
    word_start_index = torch.tensor(words.start_idx.values, dtype=torch.long)
    word_end_index = torch.tensor(words.end_idx.values, dtype=torch.long)

    # (n_words, max_seq_len, n_layers+1, hidden_size)
    # and
    # (n_words, max_seq_len, 2)
    hidden_states, offsets_mapping = compute_hidden_states(
        sentences,
        model_name,
        batch_size,
        device,
        add_special_tokens,
        return_offsets_mapping=True,
        untrained=untrained,
        revision=revision,
    )
    # Get rid of original attention masking
    hidden_states = hidden_states.get_data()

    # Get a flag for special tokens
    # (n_words, max_seq_len)
    special_tokens = offsets_mapping[:, :, 1] == 0

    # Get a mask for tokens and words correspondance
    # (n_words, max_seq_len)
    beg_tok_in_word = word_start_index[:, None] <= offsets_mapping[:, :, 0]
    end_tok_in_word = offsets_mapping[:, :, 1] <= word_end_index[:, None]

    # Aggregate into a single mask
    # (n_words, max_seq_len)
    token_word_mask = beg_tok_in_word * end_tok_in_word * ~special_tokens

    # Broadcast
    # (n_words, max_seq_len, 1, 1)
    token_word_mask = token_word_mask[:, :, None, None]
    # (n_words, max_seq_len, n_layers+1, hidden_size)
    token_word_mask = token_word_mask.broadcast_to(hidden_states.shape)
    # Build new masked tensor
    hidden_states = masked_tensor(hidden_states, token_word_mask)

    # (n_words, n_layers+1, hidden_size)
    hidden_states = aggregate_masked_tensor(
        data=hidden_states, dim=1, method=token_aggregation
    )

    # (n_words, n_layers+1, hidden_size)
    return hidden_states


class WordRepresentations(BaseModel):
    dataset: Dataset = Field(
        default_factory=lambda: Dataset(path="datasets/svo_word_level.parquet")
    )
    level: tp.Literal["word"] = "word"
    model_name: str = "bert-base-uncased"
    revision: tp.Optional[str] = None
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last"] = "mean"
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
        "device",
        "batch_size",
    )

    def model_post_init(self, context):
        super().model_post_init(context)
        assert not isinstance(self.pca, float) or self.svd_solver == "full"

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
        # (n_words, n_layers+1, hidden_size)
        return compute_word_representations(
            words=self.dataset.words_df,
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

    def __call__(self):
        import torch

        # (n_words, n_layers+1, hidden_size)
        word_representations = self.forward()

        # Subtract per-word-type mean per layer (vectorized via scatter_add)
        if self.normalize_by_word:
            words = self.dataset.words_df["word"].values
            _, inverse = np.unique(words, return_inverse=True)
            inverse_t = torch.tensor(inverse, dtype=torch.long)
            n_groups = int(inverse_t.max().item()) + 1
            n_layers, hidden_size = (
                word_representations.shape[1],
                word_representations.shape[2],
            )

            group_sums = torch.zeros(
                n_groups, n_layers, hidden_size, dtype=word_representations.dtype
            )
            counts = torch.zeros(n_groups, dtype=word_representations.dtype)
            group_sums.scatter_add_(
                0,
                inverse_t[:, None, None].expand_as(word_representations),
                word_representations,
            )
            counts.scatter_add_(
                0,
                inverse_t,
                torch.ones(len(inverse_t), dtype=word_representations.dtype),
            )
            group_means = (
                group_sums / counts[:, None, None]
            )  # (n_groups, n_layers, hidden_size)
            word_representations = word_representations - group_means[inverse_t]

        # Apply embedding normalization
        if self.normalize_embeddings == "layer0":
            # Subtract embedding layer (layer 0) from all layers
            word_representations -= word_representations[:, 0:1]
        elif self.normalize_embeddings == "diff":
            # Subtract previous layer from each layer
            word_representations[:, 1:] -= word_representations[:, :-1].clone()

        if self.units is not None:
            if isinstance(self.units, int):
                units = [self.units]
            else:
                units = self.units
            word_representations = word_representations[:, :, units]

        if self.layer is not None:
            word_representations = word_representations[:, self.layer]
        else:
            # Remove embedding layer
            word_representations = word_representations[:, 1:]
        # (n_words, hidden_size)
        n_stimuli = word_representations.shape[0]
        word_representations = word_representations.reshape(n_stimuli, -1)

        na_words = torch.isnan(word_representations).any(dim=1)
        if na_words.any():
            logger.warning(
                f"Found {na_words.sum()}/{len(na_words)} words not containing any tokens (mismatch between data and tokenizers splits). Setting their representation to 0"
            )
            word_representations[na_words] = 0

        # Apply PCA if requested (int for n_components, float for variance ratio)
        if self.pca is not None:
            from sklearn.decomposition import PCA

            original_shape = word_representations.shape
            pca_model = PCA(
                n_components=self.pca,
                random_state=self.seed,
                svd_solver=self.svd_solver,
            )
            word_representations = pca_model.fit_transform(word_representations)
            if isinstance(self.pca, float):
                logger.info(
                    f"Applied {self.pca * 100:.1f}% PCA: reduced from {original_shape[1]} to {self.pca} dimensions"
                )
            else:
                logger.info(
                    f"Applied {self.pca} components PCA: reduced from {original_shape[1]} and retained {pca_model.explained_variance_ratio_.sum() * 100:.1f}% of variance"
                )
            word_representations = torch.from_numpy(word_representations)

        scale = word_representations.std(dim=0)
        rng = np.random.default_rng(seed_from_basemodel(self))
        noise = rng.normal(scale=scale, size=word_representations.shape)
        noise *= self.noise_level

        return word_representations + torch.from_numpy(noise)
