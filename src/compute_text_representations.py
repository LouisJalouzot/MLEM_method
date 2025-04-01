from src.utils import nanmin, nanmax
from joblib.memory import Memory
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm.auto import tqdm

memory = Memory(location=".cache", verbose=0)


@memory.cache
def compute_text_representations(
    sentences,
    model_name,
    token_aggregation="mean",
    batch_size=32,
    norm=None,
    device=None,
):
    """Computes hidden states for all layers of a transformer model.

    Args:
        sentences (list): A list of sentences to process.
        model_name (str): Name of the transformer model to use.
        token_aggregation (str): How to aggregate token embeddings (mean, max, min, first, last).
        batch_size (int): Batch size for processing sentences.
        norm (int): p value for normalization. If None, no normalization is applied.
        device (torch.device): Device to use for computation. If None, uses the one defined in src.utils.

    Returns:
        torch.Tensor: Hidden states for all layers.
    """
    if device is None:
        from src.utils import device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()

    all_hidden_states = []
    for i in tqdm(
        range(0, len(sentences), batch_size),
        desc="Computing and aggregating hidden states on device " + str(device),
    ):
        batch_sentences = sentences[i : i + batch_size]
        encoded_input = tokenizer(
            batch_sentences, padding=True, truncation=False, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            hidden_states = model(**encoded_input).hidden_states

        # Tuple of torch.Tensor, one for each layer
        hidden_states = torch.stack(hidden_states)

        # Create masked hidden states with NaNs for padding tokens
        attention_mask_expanded = (
            encoded_input["attention_mask"]
            .unsqueeze(-1)
            .expand(hidden_states.size())
        )
        # Create a tensor with NaN values where attention_mask is 0
        nan_mask = torch.full_like(hidden_states, fill_value=torch.nan)
        # Use real values where attention_mask is 1, and NaNs elsewhere
        masked_hidden_states = torch.where(
            attention_mask_expanded > 0, hidden_states, nan_mask
        )

        # Aggregate token embeddings
        # hidden_states has shape (layer, batch_size, seq_len, hidden_size)
        if token_aggregation == "mean":
            aggregated_states = masked_hidden_states.nanmean(dim=2)
        elif token_aggregation == "max":
            aggregated_states = nanmax(masked_hidden_states, dim=2)[0]
        elif token_aggregation == "min":
            aggregated_states = nanmin(masked_hidden_states, dim=2)[0]
        elif token_aggregation == "first":
            aggregated_states = hidden_states[:, :, 0]
        elif token_aggregation == "last":
            # Need to handle variable sequence lengths in the batch
            last_idx = encoded_input["attention_mask"].sum(dim=1) - 1
            corresp_idx = torch.arange(len(last_idx))
            aggregated_states = hidden_states[:, corresp_idx, last_idx]
        else:
            raise ValueError(
                f"Invalid token aggregation method: {token_aggregation}"
            )

        all_hidden_states.append(aggregated_states.cpu())

    all_hidden_states = torch.concat(all_hidden_states, dim=1)

    if norm is not None:
        all_hidden_states /= all_hidden_states.norm(p=norm, dim=2, keepdim=True)

    return all_hidden_states
