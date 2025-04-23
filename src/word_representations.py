import string
import typing as tp

import pandas as pd
import torch
from exca import MapInfra
from pydantic import ConfigDict

# Removed pad_sequence
# from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

# Removed AutoModel, AutoTokenizer
# from transformers import AutoModel, AutoTokenizer

from src.utils import BaseModel  # Removed nanmax, nanmin

# Import core functions
from src.core.representations import (
    compute_word_representations,
    cum_join_index,
)

# Removed cum_join_index function


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

        # Call the core function
        yield from compute_word_representations(
            data=data,
            model_name=self.model_name,
            token_aggregation=self.token_aggregation,
            batch_size=self.batch_size,
            norm=self.norm,
            device=device,
        )

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
