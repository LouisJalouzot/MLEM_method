import string
import typing as tp

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import nanmax, nanmin


def compute_sentence_representations(
    sentences: tp.Iterable[str],
    model_name: str,
    token_aggregation: str,
    batch_size: int,
    norm: tp.Optional[int],
    device: str,
) -> tp.Iterable[torch.Tensor]:
    """Computes hidden states for all layers of a transformer model.

    Args:
        sentences: A list of sentences to process.
        model_name: Name of the Hugging Face model.
        token_aggregation: Method to aggregate token embeddings ('mean', 'max', 'min', 'first', 'last').
        batch_size: Processing batch size.
        norm: Optional normalization p-norm.
        device: Torch device ('cpu' or 'cuda:x').

    Yields:
        torch.Tensor: Hidden states for all layers for each sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()

    for i in tqdm(
        range(0, len(sentences), batch_size),
        desc=f"Computing sentence representations on device {device}",
    ):
        # Process batch of sentences
        batch_sentences = sentences[i : i + batch_size]
        encoded_input = tokenizer(
            batch_sentences,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(device)

        # Get hidden states
        with torch.no_grad():
            hidden_states = model(**encoded_input).hidden_states

        # Stack to tensor shape (layers, batch, seq_len, hidden_size)
        hidden_states = torch.stack(hidden_states)

        # Mask padding tokens with NaNs
        attention_mask_expanded = (
            encoded_input["attention_mask"]
            .unsqueeze(-1)
            .expand(hidden_states.size())
        )
        masked_hidden_states = torch.where(
            attention_mask_expanded > 0,
            hidden_states,
            torch.full_like(hidden_states, fill_value=torch.nan),
        )

        # Aggregate token embeddings based on specified method
        if token_aggregation == "mean":
            aggregated_states = masked_hidden_states.nanmean(dim=2)
        elif token_aggregation == "max":
            aggregated_states = nanmax(masked_hidden_states, dim=2)[0]
        elif token_aggregation == "min":
            aggregated_states = nanmin(masked_hidden_states, dim=2)[0]
        elif token_aggregation == "first":
            aggregated_states = hidden_states[:, :, 0]
        elif token_aggregation == "last":
            last_idx = encoded_input["attention_mask"].sum(dim=1) - 1
            corresp_idx = torch.arange(len(last_idx))
            aggregated_states = hidden_states[:, corresp_idx, last_idx]
        else:
            raise ValueError(
                f"Invalid token aggregation method: {token_aggregation}"
            )

        # Apply normalization if specified
        if norm is not None:
            aggregated_states /= aggregated_states.norm(
                p=norm, dim=2, keepdim=True
            )

        for i in range(len(batch_sentences)):
            # Yield each sentence's representation
            yield aggregated_states[:, i].cpu()  # Move to CPU before yielding


def cum_join_index(words):
    s = ""
    starts = [0]
    stops = []
    for i, w in enumerate(words):
        prefix = "" if (i == 0 or w in string.punctuation) else " "
        s += prefix + w
        stops.append(len(s))
        if i < len(words) - 1:
            # Start of next word is current length + space if needed
            next_prefix = "" if (words[i + 1] in string.punctuation) else " "
            starts.append(len(s) + len(next_prefix))
        else:
            starts.append(len(s))  # Last start is end of string

    return s.strip(), starts, stops


def compute_word_representations(
    data: tp.Iterable[tuple[str, list[int], list[int]]],
    model_name: str,
    token_aggregation: str,
    batch_size: int,
    norm: tp.Optional[int],
    device: str,
) -> tp.Iterable[torch.Tensor]:
    """Computes hidden states for all layers of a transformer model for words.

    Args:
        data: A list of tuples, each containing a sentence and its corresponding
              word start and stop indices. The tuple format is (sentence, word_start_index, word_stop_index).
        model_name: Name of the Hugging Face model.
        token_aggregation: Method to aggregate token embeddings ('mean', 'max', 'min').
        batch_size: Processing batch size.
        norm: Optional normalization p-norm.
        device: Torch device ('cpu' or 'cuda:x').

    Yields:
        torch.Tensor: Hidden states for all layers for each word.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()

    for i in tqdm(
        range(0, len(data), batch_size),
        desc=f"Computing word representations on device {device}",
    ):
        # Process batch of data
        batch = data[i : i + batch_size]
        encoded_input = tokenizer(
            [e[0] for e in batch],
            return_offsets_mapping=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(device)
        offset_mapping = encoded_input.pop("offset_mapping")

        # Get hidden states
        with torch.no_grad():
            hidden_states = model(**encoded_input).hidden_states

        # Stack to tensor shape (batch, seq_len, layer, hidden_size)
        hidden_states = torch.stack(hidden_states, dim=-2)

        # Get the word start and stop indices
        word_start_index = [torch.Tensor(e[1]) for e in batch]
        word_start_index = pad_sequence(word_start_index, batch_first=True)
        word_start_index = word_start_index.to(device)
        word_stop_index = [torch.Tensor(e[2]) for e in batch]
        word_stop_index = pad_sequence(word_stop_index, batch_first=True)
        word_stop_index = word_stop_index.to(device)

        # Get a flag for special tokens
        special_tokens = offset_mapping[:, :, 1] == offset_mapping[:, :, 0]

        # Get a mask for tokens and words correspondance
        beg_tok_in_word = (
            word_start_index[:, :, None] <= offset_mapping[:, None, :, 0]
        )
        end_tok_in_word = (
            offset_mapping[:, None, :, 1] <= word_stop_index[:, :, None]
        )
        token_word_mask = (
            beg_tok_in_word * end_tok_in_word * ~special_tokens[:, None]
        )

        # Broadcast and filter
        hidden_states = hidden_states[:, None].repeat(
            1, word_start_index.shape[1], 1, 1, 1
        )
        hidden_states[~token_word_mask] = torch.nan

        if token_aggregation == "mean":
            aggregated_states = hidden_states.nanmean(dim=2)
        elif token_aggregation == "max":
            aggregated_states = nanmax(hidden_states, dim=2)[0]
        elif token_aggregation == "min":
            aggregated_states = nanmin(hidden_states, dim=2)[0]
        elif token_aggregation == "first":
            raise NotImplementedError
        elif token_aggregation == "last":
            raise NotImplementedError
        else:
            raise ValueError(
                f"Unknown token aggregation method: {token_aggregation}"
            )

        # Apply normalization if specified
        if norm is not None:
            aggregated_states /= aggregated_states.norm(
                p=norm, dim=-1, keepdim=True
            )

        for aggregated_state in aggregated_states:
            # Yield each sentence's representation
            yield aggregated_state.cpu()
