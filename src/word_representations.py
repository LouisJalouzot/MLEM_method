import string
import typing as tp

import pandas as pd
import torch
from exca import TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from torch.masked import masked_tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig

from src.dataset import Dataset
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
        # (n_words, n_layers+1, hidden_size)
        word_representations = self.forward()
        if self.units is not None:
            word_representations = word_representations[:, :, self.units]

        # (n_words, hidden_size)
        word_representations = word_representations[:, self.layer]
        na_words = torch.isnan(word_representations).any(dim=1)
        if na_words.any():
            logger.warning(
                f"Found {na_words.sum()}/{len(na_words)} words not containing any tokens (mismatch between data and tokenizers splits). Setting their representation to 0"
            )
            word_representations[na_words] = 0

        return word_representations

    @infra.apply(exclude_from_cache_uid=["layer", "units"])
    def forward(self) -> torch.Tensor:
        # (n_words, n_layers+1, hidden_size)
        return compute_word_representations(
            words=self.dataset.words_df,
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            add_special_tokens=self.add_special_tokens,
            token_aggregation=self.token_aggregation,
        )
