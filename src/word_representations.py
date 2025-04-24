import string
import typing as tp

import pandas as pd
import torch
from exca import TaskInfra
from pydantic import ConfigDict
from torch.masked import masked_tensor
from torch.nn.utils.rnn import pad_sequence

from src.hidden_states import aggregate_masked_tensor, compute_hidden_states
from src.utils import BaseModel


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
    sentences, word_start_index, word_stop_index, word_mask = [], [], [], []
    for _, group in words.groupby("sentence_id"):
        group_words = group.word.tolist()
        word_mask.append(torch.full((len(group_words),), True))
        sentence, starts, stops = cum_join_index(group_words)
        sentences.append(sentence)
        word_start_index.append(torch.Tensor(starts))
        word_stop_index.append(torch.Tensor(stops))

    # (total_sentences, max_words)
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
    hidden_states = hidden_states.get_data()

    # Get a flag for special tokens
    # (total_sentences, max_seq_len)
    special_tokens = offsets_mapping[:, :, 0] == 0

    # Get a mask for tokens and words correspondance
    # (total_sentences, max_words + 1, max_seq_len)
    beg_tok_in_word = (
        word_start_index[:, :, None] <= offsets_mapping[:, None, :, 0]
    )
    end_tok_in_word = (
        offsets_mapping[:, None, :, 1] <= word_stop_index[:, :, None]
    )

    # Aggregate into a single mask
    # (total_sentences, max_words + 1, max_seq_len)
    token_word_mask = (
        beg_tok_in_word * end_tok_in_word * ~special_tokens[:, None]
    )

    # Broadcast
    # (layers, total_sentences, 1, max_seq_len, hidden_size)
    hidden_states = hidden_states[:, :, None]
    # (1, total_sentences, max_words, max_seq_len, 1)
    token_word_mask = token_word_mask[None, ..., None]
    # (layers, total_sentences, max_words, max_seq_len, hidden_size)
    hidden_states, token_word_mask = torch.broadcast_tensors(
        hidden_states, token_word_mask
    )
    # Build new masked tensor
    hidden_states = masked_tensor(hidden_states, token_word_mask)

    # (layers, total_sentences, max_words, hidden_size)
    hidden_states = aggregate_masked_tensor(
        data=hidden_states, dim=3, method=token_aggregation
    )

    # (layers, total_words, hidden_size)
    return hidden_states[:, word_mask]


# class WordRepresentations(BaseModel):
#     level: tp.Literal["word"] = "word"
#     model_name: str = "bert-base-uncased"
#     token_aggregation: str = "mean"
#     batch_size: int = 32
#     layer: int = 5
#     units: tp.List[int] = None
#     norm: tp.Optional[int] = None
#     _device: tp.Optional[str] = None
#     infra: TaskInfra = TaskInfra(folder=".cache")
#     model_config: ConfigDict = ConfigDict(extra="forbid")

#     def __call__(self) -> torch.Tensor:
#         pass

#     @infra.apply(item_uid=str, exclude_from_cache_uid=["layer", "units"])
#     def _compute_word_representations(self):
#         pass
