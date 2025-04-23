import typing as tp

import torch
from exca import MapInfra
from pydantic import ConfigDict
from tqdm.auto import tqdm

# Removed AutoModel, AutoTokenizer imports
# from transformers import AutoModel, AutoTokenizer

from src.utils import BaseModel  # Removed nanmax, nanmin

# Import core function
from src.core.representations import compute_sentence_representations


class SentenceRepresentations(BaseModel):
    level: tp.Literal["sentence"] = "sentence"
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
        self, sentences: tp.Iterable[str]
    ) -> tp.Iterable[torch.Tensor]:
        """Computes hidden states for all layers of a transformer model.

        Args:
            sentences: A list of sentences to process.

        Returns:
            torch.Tensor: Hidden states for all layers.
        """
        if self._device is None:
            from src.utils import device
        else:
            device = self._device

        # Call the core function
        yield from compute_sentence_representations(
            sentences=sentences,
            model_name=self.model_name,
            token_aggregation=self.token_aggregation,
            batch_size=self.batch_size,
            norm=self.norm,
            device=device,
        )

    def compute_representations(
        self,
        sentences: tp.Iterable[str],
    ) -> torch.Tensor:
        representations = []
        for repr in tqdm(
            self._compute_representations_cached(sentences),
            desc=f"Retrieving representations",
            total=len(sentences),
        ):
            repr = repr[self.layer]
            if self.units is not None:
                repr = repr[self.units]
            representations.append(repr)
        return torch.stack(representations)
