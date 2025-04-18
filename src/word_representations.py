import string
import typing as tp

import pandas as pd
import torch
from exca import MapInfra
from pydantic import ConfigDict
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import BaseModel, nanmax, nanmin


def cum_join_index(words):
    s = ""
    starts = [0]
    stops = []
    for w in words:
        if w not in string.punctuation and len(s) > 0:
            s += " "
        s += w
        n = len(s)
        stops.append(n)
        starts.append(n)
    stops.append(len(s))

    return s.strip(), starts, stops


class WordRepresentations(BaseModel):
    level: tp.Literal["word"] = "word"
    model_name: str = "bert-base-uncased"
    token_aggregation: str = "mean"
    batch_size: int = 32
    layer: int = 5
    units: tp.List[int] = None
    norm: tp.Optional[int] = None
    _device: tp.Optional[str] = None
    infra: MapInfra = MapInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @infra.apply(item_uid=str, exclude_from_cache_uid=["layer", "units"])
    def _compute_representations_cached(
        self, data: tp.Iterable[tuple[str, list[int], list[int]]]
    ) -> tp.Iterable[torch.Tensor]:
        """Computes hidden states for all layers of a transformer model.

        Args:
            data: A list of tuples, each containing a sentence and its corresponding
                  word start and stop indices. The tuple format is (sentence, word_start_index, word_stop_index).

        Returns:
            torch.Tensor: Hidden states for all layers.
        """
        if self._device is None:
            from src.utils import device
        else:
            device = self._device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            self.model_name, output_hidden_states=True
        )
        model = model.to(device)
        model.eval()

        for i in tqdm(
            range(0, len(data), self.batch_size),
            desc=f"Computing representations on device {device}",
        ):
            # Process batch of data
            batch = data[i : i + self.batch_size]
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

            if self.token_aggregation == "mean":
                aggregated_states = hidden_states.nanmean(dim=2)
            elif self.token_aggregation == "max":
                aggregated_states = nanmax(hidden_states, dim=2)[0]
            elif self.token_aggregation == "min":
                aggregated_states = nanmin(hidden_states, dim=2)[0]
            elif self.token_aggregation == "first":
                raise NotImplementedError
            elif self.token_aggregation == "last":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown token aggregation method: {self.token_aggregation}"
                )

            # Apply normalization if specified
            if self.norm is not None:
                aggregated_states /= aggregated_states.norm(
                    p=self.norm, dim=-1, keepdim=True
                )

            for aggregated_state in aggregated_states:
                # Yield each sentence's representation
                yield aggregated_state.cpu()

    def compute_representations(self, words: pd.DataFrame) -> torch.Tensor:
        sentences = words.groupby("sentence_id").word.apply(cum_join_index)
        sentences = sentences.tolist()

        out = []
        for t in tqdm(
            self._compute_representations_cached(sentences),
            total=len(sentences),
            desc="Retrieving sentence representations",
        ):
            t = t[:, self.layer]
            if self.units is not None:
                t = t[:, self.units]
            out.append(t)
        out = torch.concat(out, dim=0)
        # Remove padding
        out = out[~torch.isnan(out).all(dim=1)]

        return out
