import string
import typing as tp
from typing import Any

import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict
from torch.masked import masked_tensor
from torch.nn.utils.rnn import pad_sequence

from src.hidden_states import aggregate_masked_tensor, compute_hidden_states
from src.utils import BaseModel, get_device


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
) -> torch.Tensor:
    """Computes word representations from hidden states by aggregating token representations.

    Args:
        words (pd.DataFrame): DataFrame containing columns 'word' (str) and
            'sentence_id' (int). Each row represents a word. Words belonging
            to the same sentence must have the same 'sentence_id' and will be
            joined into sentences in the order of their appearance in the
            DataFrame. Assumes English punctuation rules (no space before
            punctuation marks like ',', '.', '?').
        model_name (str, optional): Name of the Hugging Face model to use.
            Defaults to "prajjwal1/bert-tiny".
        token_aggregation (str, optional): Method to aggregate token representations
            ('mean', 'sum', etc.). Defaults to "mean".
        batch_size (int, optional): Batch size for processing sentences. Defaults to 32.
        device (str, optional): Device to run computations on ('cpu', 'cuda').
            Defaults to "cpu".
        add_special_tokens (bool, optional): Whether to include special tokens
            ([CLS], [SEP]) during tokenization. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing word representations for all words across
            all sentences, shaped (n_words, layers, hidden_size).

    Example:
        The following will process two sentences:
        "This is an example." and "Another sentence!"
        ```python
        import pandas as pd
        words_df = pd.DataFrame({
            'word': ['This', 'is', 'an', 'example', '.', 'Another', 'sentence', '!'],
            'sentence_id': [0, 0, 0, 0, 0, 1, 1, 1]
        })
        representations = compute_word_representations(words_df)
        ```
    """
    sentences, word_start_index, word_stop_index, word_mask = [], [], [], []
    for _, group in words.groupby("sentence_id"):
        group_words = group.word.tolist()
        word_mask.append(torch.full((len(group_words),), True))
        sentence, starts, stops = cum_join_index(group_words)
        sentences.append(sentence)
        word_start_index.append(torch.Tensor(starts))
        word_stop_index.append(torch.Tensor(stops))

    # (n_sentences, max_words)
    word_start_index = pad_sequence(word_start_index, batch_first=True)
    word_stop_index = pad_sequence(word_stop_index, batch_first=True)
    word_mask = pad_sequence(word_mask, batch_first=True, padding_value=False)

    hidden_states, offsets_mapping = compute_hidden_states(
        sentences,
        model_name,
        batch_size,
        device,
        add_special_tokens,
        return_offsets_mapping=True,
    )
    # Get rid of original attention masking
    # (n_sentences, max_seq_len, n_layers+1, hidden_size)
    hidden_states = hidden_states.get_data()

    # Get a flag for special tokens
    # (n_sentences, max_seq_len)
    special_tokens = offsets_mapping[:, :, 0] == 0

    # Get a mask for tokens and words correspondance
    # (n_sentences, max_words, max_seq_len)
    beg_tok_in_word = (
        word_start_index[:, :, None] <= offsets_mapping[:, None, :, 0]
    )
    end_tok_in_word = (
        offsets_mapping[:, None, :, 1] <= word_stop_index[:, :, None]
    )

    # Aggregate into a single mask
    # (n_sentences, max_words, max_seq_len)
    token_word_mask = (
        beg_tok_in_word * end_tok_in_word * ~special_tokens[:, None]
    )

    # Broadcast
    # (n_sentences, 1, max_seq_len, n_layers+1, hidden_size)
    hidden_states = hidden_states[:, None]
    # (n_sentences, max_words, max_seq_len, 1, 1)
    token_word_mask = token_word_mask[:, :, :, None, None]
    # (n_sentences, max_words, max_seq_len, n_layers+1, hidden_size)
    hidden_states, token_word_mask = torch.broadcast_tensors(
        hidden_states, token_word_mask
    )
    # Build new masked tensor
    hidden_states = masked_tensor(hidden_states, token_word_mask)

    # (n_sentences, max_words, n_layers+1, hidden_size)
    hidden_states = aggregate_masked_tensor(
        data=hidden_states, dim=3, method=token_aggregation
    )

    # (n_words, n_layers+1, hidden_size)
    return hidden_states[word_mask]


class WordRepresentations(BaseModel):
    """Computes word representations using a pre-trained transformer model.

    This class handles the computation of word embeddings by processing words grouped
    by sentence through a specified Hugging Face transformer model. It aligns token
    representations to words and allows configuration of model name, batch size,
    token aggregation method, and selection of specific layers and units. Caching is
    used to avoid recomputing representations for the same configuration (excluding
    layer and unit selection).

    Attributes:
        words: A list of strings, where each string is a word.
        sentence_id: A list of integers mapping each word in `words` to its sentence.
            Words with the same ID belong to the same sentence and are processed together
            in their order of appearance.
        level: The level of representation, fixed to "word".
        model_name: The name of the Hugging Face model to use (e.g., "bert-base-uncased").
        token_aggregation: The method used to aggregate token hidden states corresponding
            to a word into a single word representation ("mean", "max", "min", "first", "last").
        add_special_tokens: Whether to include special tokens ([CLS], [SEP]) during
            tokenization. Affects offset mapping used for word alignment.
        batch_size: The batch size used for processing sentences through the model.
        layer: The specific layer from which to extract the final word representations.
        units: An optional list of specific hidden unit indices to select from the
            chosen layer's representation. If None, all units are kept.
        infra: Configuration for caching mechanism.
        model_config: Pydantic model configuration.
    """

    words: tp.List[str]
    sentence_id: tp.List[int]
    level: tp.Literal["word"] = "word"
    model_name: str = "bert-base-uncased"
    token_aggregation: tp.Literal["mean", "max", "min", "first", "last"] = (
        "mean"
    )
    add_special_tokens: bool = True
    batch_size: int = 32
    layer: int = 5
    units: tp.List[int] = None
    _device: tp.Optional[torch.device] = None
    infra: TaskInfra = TaskInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """Sets the device for computation if not provided."""
        if self._device is None:
            self._device = get_device()

    @infra.apply(exclude_from_cache_uid=["layer", "units"])
    def _compute_representations_cached(self):
        words = pd.DataFrame(
            {"word": self.words, "sentence_id": self.sentence_id}
        )

        # (n_words, layers, hidden_size)
        return compute_word_representations(
            words,
            model_name=self.model_name,
            token_aggregation=self.token_aggregation,
            batch_size=self.batch_size,
            device=self._device,
            add_special_tokens=self.add_special_tokens,
        )

    def __call__(self) -> torch.Tensor:
        # (n_words, layers, hidden_size)
        word_representations = self._compute_representations_cached()
        if self.units is not None:
            word_representations = word_representations[:, :, self.units]

        # (n_words, hidden_size)
        return word_representations[:, self.layer]
