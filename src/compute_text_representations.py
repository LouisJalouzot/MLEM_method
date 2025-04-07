from src.utils import nanmin, nanmax
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm.auto import tqdm
from src.utils import BaseModel, ConfigDict
from exca import MapInfra
import typing as tp


class TextRepresentations(BaseModel):
    model_name: str = "bert-base-uncased"
    token_aggregation: str = "mean"
    batch_size: int = 32
    norm: tp.Optional[int] = None
    device: tp.Optional[str] = None
    infra: MapInfra = MapInfra(version="1", folder=".cache")

    model_config: ConfigDict = ConfigDict(extra="forbid")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    @infra.apply(item_uid=str)
    def compute_representations(
        self, sentences: tp.Iterable[str]
    ) -> tp.Iterable[torch.Tensor]:
        """Computes hidden states for all layers of a transformer model.

        Args:
            sentences: A list of sentences to process.

        Returns:
            torch.Tensor: Hidden states for all layers.
        """
        if self.device is None:
            from src.utils import device
        else:
            device = self.device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            self.model_name, output_hidden_states=True
        )
        model = model.to(device)
        model.eval()

        for i in tqdm(
            range(0, len(sentences), self.batch_size),
            desc=f"Computing representations on device {device}",
        ):
            # Process batch of sentences
            batch_sentences = sentences[i : i + self.batch_size]
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
            if self.token_aggregation == "mean":
                aggregated_states = masked_hidden_states.nanmean(dim=2)
            elif self.token_aggregation == "max":
                aggregated_states = nanmax(masked_hidden_states, dim=2)[0]
            elif self.token_aggregation == "min":
                aggregated_states = nanmin(masked_hidden_states, dim=2)[0]
            elif self.token_aggregation == "first":
                aggregated_states = hidden_states[:, :, 0]
            elif self.token_aggregation == "last":
                last_idx = encoded_input["attention_mask"].sum(dim=1) - 1
                corresp_idx = torch.arange(len(last_idx))
                aggregated_states = hidden_states[:, corresp_idx, last_idx]
            else:
                raise ValueError(
                    f"Invalid token aggregation method: {self.token_aggregation}"
                )

            # Apply normalization if specified
            if self.norm is not None:
                aggregated_states /= aggregated_states.norm(
                    p=self.norm, dim=2, keepdim=True
                )

            for i in range(len(batch_sentences)):
                # Yield each sentence's representation
                yield aggregated_states[:, i]

    def compute_and_combine_representations(
        self, sentences: tp.Iterable[str]
    ) -> torch.Tensor:
        representations = []
        for repr in tqdm(
            self.compute_representations(sentences),
            desc=f"Retrieving representations",
            total=len(sentences),
        ):
            representations.append(repr)
        return torch.stack(representations, dim=1)
