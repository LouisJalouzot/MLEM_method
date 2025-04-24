import typing as tp
import warnings

import torch
from torch.masked import MaskedTensor, masked_tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)


def compute_hidden_states(
    sentences: tp.List[str],
    model_name: str = "prajjwal1/bert-tiny",
    batch_size: int = 32,
    device: str = "cpu",
    add_special_tokens: bool = True,
    return_offsets_mapping: bool = False,
) -> MaskedTensor | tp.Tuple[MaskedTensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()

    # Tokenize all sentences at once
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=return_offsets_mapping,
    )

    # (total_sentences, max_seq_len)
    attention_mask = encoded_input["attention_mask"]
    if return_offsets_mapping:
        # (total_sentences, max_seq_len, 2)
        offsets_mapping = encoded_input.pop("offset_mapping")
    # (total_sentences, max_seq_len)
    input_ids = encoded_input["input_ids"]

    hidden_states = []
    for i in tqdm(
        range(0, len(sentences), batch_size),
        desc=f"Computing sentence representations on device {device}",
    ):
        batch_input_ids = input_ids[i : i + batch_size].to(device)
        batch_attention_mask = attention_mask[i : i + batch_size].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
            batch_hidden_states = outputs.hidden_states

        # Stack to tensor shape (layers, batch, seq_len, hidden_size)
        batch_hidden_states = torch.stack(batch_hidden_states)

        hidden_states.append(batch_hidden_states.cpu())

    # (layers, total_sentences, max_seq_len, hidden_size)
    hidden_states = torch.cat(hidden_states, dim=1)

    # Broadcast attention mask to match hidden states shape and cast to bool
    # (1, total_sentences, max_seq_len, 1)
    attention_mask = attention_mask[None, ..., None]
    # (total_sentences, layers, max_seq_len, hidden_size)
    attention_mask = attention_mask.broadcast_to(hidden_states.shape)
    # Mask hidden states based on the full attention mask
    all_hidden_states_masked = masked_tensor(
        hidden_states, attention_mask.bool()
    )

    if return_offsets_mapping:
        return all_hidden_states_masked, offsets_mapping
    else:
        return all_hidden_states_masked


def aggregate_masked_tensor(
    data: MaskedTensor, dim: int, method: str = "mean"
) -> torch.Tensor:
    if method == "mean":
        return data.mean(dim=dim).get_data()
    elif method == "min":
        return data.amin(dim=dim).get_data()
    elif method == "max":
        return data.amax(dim=dim).get_data()
    elif method == "first":
        mask = data.get_mask().int()
        first_idx = mask.argmax(dim=dim, keepdim=True)
        agg = data.get_data().gather(dim, first_idx)
        return agg.select(dim, 0)
    elif method == "last":
        mask = data.get_mask().int()
        max_seq_len = mask.shape[dim]
        last_idx = (
            max_seq_len
            - mask.flip(dims=(dim,)).argmax(dim=dim, keepdim=True)
            - 1
        )
        agg = data.get_data().gather(dim, last_idx)
        return agg.select(dim, 0)
    else:
        raise ValueError(
            f"Unsupported aggregation method: {method}. "
            "Choose from 'mean', 'min', 'max', 'first', 'last'."
        )
