from __future__ import annotations

import typing as tp
import warnings

if tp.TYPE_CHECKING:
    import torch
    from torch.masked import MaskedTensor

from tqdm.auto import tqdm

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
    """
    Computes hidden states for a list of sentences using a specified transformer model.

    Args:
        sentences: A list of strings, where each string is a sentence.
        model_name: The name of the pre-trained transformer model to use
            (e.g., 'bert-base-uncased', 'prajjwal1/bert-tiny').
        batch_size: The number of sentences to process in each batch.
        device: The device to run the model on ('cpu' or 'cuda').
        add_special_tokens: Whether to add special tokens (like [CLS], [SEP])
            during tokenization. Defaults to True.
        return_offsets_mapping: Whether to return the character offsets mapping
            for each token. Defaults to False.

    Returns:
        If `return_offsets_mapping` is False:
            A MaskedTensor of shape (n_sentences, max_seq_len, n_layers+1, hidden_size)
            containing the hidden states for each token in each sentence across all layers.
            Padding tokens are masked.
        If `return_offsets_mapping` is True:
            A tuple containing:
            - The MaskedTensor of hidden states as described above.
            - A torch.Tensor of shape (n_sentences, max_seq_len, 2) containing the
              start and end character offsets for each token.

    Example:
        >>> sentences = ["This is a test sentence.", "Another example."]
        >>> hidden_states_masked = compute_hidden_states(sentences, model_name='prajjwal1/bert-tiny')
        >>> print(hidden_states_masked.shape)
        torch.Size([2, 8, 3, 128]) # (n_sentences, max_seq_len, n_layers+1, hidden_size)
        >>> print(hidden_states_masked.get_mask().shape)
        torch.Size([2, 8, 3, 128])

        >>> hidden_states_masked, offsets = compute_hidden_states(
        ...     sentences, model_name='prajjwal1/bert-tiny', return_offsets_mapping=True
        ... )
        >>> print(hidden_states_masked.shape)
        torch.Size([2, 8, 3, 128])
        >>> print(offsets.shape)
        torch.Size([2, 8, 2])
    """
    import torch
    from torch.masked import masked_tensor
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    # For encoder-decoder models (T5, BART, etc.), use only the encoder
    # to extract hidden states for sentence representations
    is_encoder_decoder = hasattr(model, "encoder") and hasattr(model, "decoder")
    if is_encoder_decoder:
        model = model.encoder

    # Tokenize all sentences at once
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=return_offsets_mapping,
    )

    # (n_sentences, max_seq_len)
    attention_mask = encoded_input["attention_mask"]
    if return_offsets_mapping:
        # (n_sentences, max_seq_len, 2)
        offsets_mapping = encoded_input.pop("offset_mapping")
    # (n_sentences, max_seq_len)
    input_ids = encoded_input["input_ids"]

    hidden_states = []  # Renamed for clarity
    with tqdm(
        desc=f"Computing sentence representations on device {device}",
        total=len(sentences),
    ) as pbar:
        for i in range(0, len(sentences), batch_size):
            batch_input_ids = input_ids[i : i + batch_size].to(device)
            batch_attention_mask = attention_mask[i : i + batch_size].to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.split(":")[0]):
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
                    batch_hidden_states = outputs.hidden_states

            # Stack to tensor shape (n_layers+1, batch, max_seq_len, hidden_size)
            batch_hidden_states = torch.stack(batch_hidden_states)

            # Permute to (batch, max_seq_len, n_layers+1, hidden_size)
            batch_hidden_states_permuted = batch_hidden_states.permute(1, 2, 0, 3)

            hidden_states.append(batch_hidden_states_permuted.cpu())

            pbar.update(batch_input_ids.shape[0])

    # Concatenate along batch dimension (dim=0)
    # Resulting shape: (n_sentences, max_seq_len, n_layers+1, hidden_size)
    hidden_states = torch.cat(hidden_states, dim=0)

    # Broadcast attention mask to match hidden states shape and cast to bool
    # (n_sentences, max_seq_len) -> (n_sentences, max_seq_len, 1, 1)
    attention_mask = attention_mask[:, :, None, None]
    # Broadcast to (n_sentences, max_seq_len, n_layers+1, hidden_size)
    attention_mask = attention_mask.broadcast_to(hidden_states.shape)

    # Mask hidden states based on the full attention mask
    all_hidden_states_masked = masked_tensor(hidden_states, attention_mask.bool())

    if return_offsets_mapping:
        return all_hidden_states_masked, offsets_mapping
    else:
        return all_hidden_states_masked


def aggregate_masked_tensor(
    data: MaskedTensor, dim: int, method: str = "mean"
) -> torch.Tensor:
    """
    Aggregates a MaskedTensor along a specified dimension, ignoring masked values.

    Args:
        data: The input MaskedTensor.
        dim: The dimension along which to aggregate.
        method: The aggregation method to use. Supported methods are:
            'mean': Computes the mean of unmasked values.
            'min': Computes the minimum of unmasked values.
            'max': Computes the maximum of unmasked values.
            'first': Selects the first unmasked value along the dimension.
            'last': Selects the last unmasked value along the dimension.
            Defaults to 'mean'.

    Returns:
        A torch.Tensor containing the aggregated values. The specified dimension `dim`
        is removed.

    Raises:
        ValueError: If an unsupported aggregation method is provided.

    Example:
        >>> data_tensor = torch.tensor([[1., 2., 3.], [4., 5., 0.]])
        >>> mask_tensor = torch.tensor([[True, True, True], [True, True, False]])
        >>> masked_data = masked_tensor(data_tensor, mask_tensor)
        >>> print(masked_data)
        MaskedTensor(
            [
                [  1.0000,       --,   3.0000],
                [  4.0000,   5.0000,       --]
            ]
        )
        >>> aggregate_masked_tensor(masked_data, dim=0, method='mean')
        tensor([2.5000, 5.0000, 3.0000])
        >>> aggregate_masked_tensor(masked_data, dim=0, method='max')
        tensor([4., 5., 3.])
        >>> aggregate_masked_tensor(masked_data, dim=0, method='first')
        tensor([1., 5., 3.])
        >>> aggregate_masked_tensor(masked_data, dim=0, method='last')
        tensor([4., 5., 3.])
    """
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
            max_seq_len - mask.flip(dims=(dim,)).argmax(dim=dim, keepdim=True) - 1
        )
        agg = data.get_data().gather(dim, last_idx)
        return agg.select(dim, 0)
    else:
        raise ValueError(
            f"Unsupported aggregation method: {method}. "
            "Choose from 'mean', 'min', 'max', 'first', 'last'."
        )
