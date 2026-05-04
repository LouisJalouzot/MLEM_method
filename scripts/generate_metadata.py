import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, model_info
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def get_first_attr(obj, *names):
    for name in names:
        value = obj.get(name)
        if value is not None:
            return value
    raise AttributeError(f"Could not find any of {names} in config")


def get_attention_type(config, architecture):
    if architecture != "transformer":
        return np.nan

    n_heads = config.get("num_attention_heads") or config.get("n_head")
    n_kv_heads = config.get("num_key_value_heads") or n_heads

    if n_heads is None:
        return np.nan
    if n_kv_heads == n_heads:
        return "MHA"
    if n_kv_heads == 1:
        return "MQA"
    return "GQA"


def get_activation_type(config):
    act = config.get("hidden_act") or config.get("activation_function")
    if not act and "mamba" in config.get("model_type", "").lower():
        act = "silu"

    if act == "gelu_new":
        act = "gelu"

    return act if act else np.nan


def get_normalization_type(config):
    if "rms_norm_eps" in config or config.get("norm_type") == "rmsnorm":
        return "RMSNorm"
    if "layer_norm_epsilon" in config or "layer_norm_eps" in config or config.get("do_layer_norm_before") is not None:
        return "LayerNorm"
    return "LayerNorm"


def get_positional_encoding(config, architecture):
    if (
        "rope_theta" in config
        or "rope_scaling" in config
        or "rope_parameters" in config
        or config.get("position_embedding_type") == "rope"
        or "rotary_emb_base" in config
    ):
        return "RoPE"
    if config.get("position_embedding_type") == "alibi" or config.get("alibi"):
        return "ALiBi"
    if "mamba" in architecture.lower() or "rwkv" in architecture.lower():
        return np.nan
    return "Absolute"


def get_model_metadata(model_id, architecture, n_params_B, n_tokens_B):
    info = model_info(model_id)
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config = config.get("text_config", config)

    # Depth/Width/Vocab from config
    n_layers = get_first_attr(config, "num_hidden_layers", "n_layer", "num_layers")
    hidden_size = get_first_attr(config, "hidden_size", "n_embd", "d_model")

    release_date = info.created_at.year + (info.created_at.month - 1) / 12

    return [
        model_id,
        architecture,
        get_attention_type(config, architecture),
        get_activation_type(config),
        get_normalization_type(config),
        get_positional_encoding(config, architecture),
        np.log10(n_params_B),
        release_date,
        np.log2(n_layers),
        n_layers / hidden_size,
        np.log10(n_tokens_B),
    ]


def generate_metadata():
    # model_id, architecture, n_params_B, n_tokens_B
    models_to_fetch = [
        ("openai-community/gpt2", "transformer", 0.117, 10),  # Estimated
        ("openai-community/gpt2-medium", "transformer", 0.345, 10),  # Estimated
        ("openai-community/gpt2-large", "transformer", 0.762, 10),  # Estimated
        ("openai-community/gpt2-xl", "transformer", 1.5, 10),  # Estimated
        ("facebook/opt-125m", "transformer", 0.125, 180),
        ("facebook/opt-1.3b", "transformer", 1.3, 180),
        ("facebook/opt-2.7b", "transformer", 2.7, 180),
        ("facebook/opt-6.7b", "transformer", 6.7, 180),
        ("facebook/opt-13b", "transformer", 13.0, 180),
        ("EleutherAI/pythia-410m-deduped", "transformer", 0.410, 207),
        ("EleutherAI/pythia-1b-deduped", "transformer", 1.0, 207),
        ("EleutherAI/pythia-1.4b-deduped", "transformer", 1.4, 207),
        ("EleutherAI/pythia-6.9b-deduped", "transformer", 6.9, 207),
        ("EleutherAI/pythia-12b-deduped", "transformer", 12.0, 207),
        ("allenai/OLMo-2-0425-1B", "transformer", 1.0, 4000),
        ("allenai/OLMo-2-1124-7B", "transformer", 7.0, 4000),
        ("allenai/OLMo-2-1124-13B", "transformer", 13.0, 5000),
        ("meta-llama/Llama-3.2-1B", "transformer", 1.0, 9000),
        ("meta-llama/Llama-3.2-3B", "transformer", 3.0, 9000),
        ("meta-llama/Llama-3.1-8B", "transformer", 8.0, 15000),
        ("mistralai/Ministral-3-3B-Base-2512", "transformer", 3.0, 3000),
        ("mistralai/Ministral-3-8B-Base-2512", "transformer", 8.0, 3000),
        ("mistralai/Ministral-3-14B-Base-2512", "transformer", 14.0, 3000),
        ("Qwen/Qwen3-0.6B-Base", "transformer", 0.6, 36000),
        ("Qwen/Qwen3-1.7B-Base", "transformer", 1.7, 36000),
        ("Qwen/Qwen3-4B-Base", "transformer", 4.0, 36000),
        ("Qwen/Qwen3-8B-Base", "transformer", 8.0, 36000),
        ("Qwen/Qwen3-14B-Base", "transformer", 14.0, 36000),
        ("state-spaces/mamba-130m-hf", "mamba", 0.130, 300),
        ("state-spaces/mamba-370m-hf", "mamba", 0.370, 300),
        ("state-spaces/mamba-790m-hf", "mamba", 0.790, 300),
        ("state-spaces/mamba-1.4b-hf", "mamba", 1.4, 300),
        ("state-spaces/mamba-2.8b-hf", "mamba", 2.8, 300),
        ("AntonV/mamba2-130m-hf", "mamba", 0.130, 300),
        ("AntonV/mamba2-370m-hf", "mamba", 0.370, 300),
        ("AntonV/mamba2-780m-hf", "mamba", 0.780, 300),
        ("AntonV/mamba2-1.3b-hf", "mamba", 1.3, 300),
        ("AntonV/mamba2-2.7b-hf", "mamba", 2.7, 300),
        ("fla-hub/rwkv7-191M-world", "rwkv", 0.191, 1600),
        ("fla-hub/rwkv7-0.4B-world", "rwkv", 0.4, 3100),
        ("fla-hub/rwkv7-1.5B-world", "rwkv", 1.5, 5600),
        ("fla-hub/rwkv7-2.9B-world", "rwkv", 2.9, 5600),
        ("fla-hub/rwkv7-7.2B-g0a", "rwkv", 7.2, 5600),
    ]

    results = tqdm(
        Parallel(n_jobs=-2, return_as="generator", backend="threading")(
            delayed(get_model_metadata)(m_id, arch, n_params_B, n_tokens_B)
            for m_id, arch, n_params_B, n_tokens_B in models_to_fetch
        ),
        total=len(models_to_fetch),
        desc="Fetching model metadata",
    )

    columns = [
        "model_name",
        "Architecture class",
        "Attention Type",
        "Activation Type",
        "Normalization",
        "Positional Encoding",
        "Num. Parameters",
        "Release Date",
        "Depth",
        "Depth / Width",
        "Training Tokens",
    ]
    df = pd.DataFrame(results, columns=columns)
    output_path = Path(__file__).parent.parent / "model_metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    generate_metadata()
