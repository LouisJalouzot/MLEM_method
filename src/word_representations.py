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
            all sentences, shaped (n_words, n_layers+1, hidden_size).

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
    special_tokens = offsets_mapping[:, :, 1] == 0

    # Get a mask for tokens and words correspondance
    # (n_sentences, max_words, max_seq_len)
    beg_tok_in_word = word_start_index[:, :, None] <= offsets_mapping[:, None, :, 0]
    end_tok_in_word = offsets_mapping[:, None, :, 1] <= word_stop_index[:, :, None]

    # Aggregate into a single mask
    # (n_sentences, max_words, max_seq_len)
    token_word_mask = beg_tok_in_word * end_tok_in_word * ~special_tokens[:, None]

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
        data=hidden_states, dim=2, method=token_aggregation
    )

    # (n_words, n_layers+1, hidden_size)
    return hidden_states[word_mask]


class WordRepresentations(BaseModel):
    dataset: Dataset = Field(
        default_factory=lambda: Dataset(csv_path="datasets/svo_word_level.csv")
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
        sentence_representations = self.forward()
        if self.units is not None:
            sentence_representations = sentence_representations[:, :, self.units]

        # (n_words, hidden_size)
        return sentence_representations[:, self.layer]

    @infra.apply(exclude_from_cache_uid=["layer", "units"])
    def forward(self) -> torch.Tensor:
        raise NotImplementedError()
        # TODO: Now data has word, sentence, start_idx, end_idx: need to change
        words = pd.DataFrame(
            {
                "word": self.dataset.words,
                "sentence_id": self.dataset.sentence_id,
            }
        )

        # (n_words, n_layers+1, hidden_size)
        return compute_word_representations(
            words=words,
            model_name=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            add_special_tokens=self.add_special_tokens,
            token_aggregation=self.token_aggregation,
        )
