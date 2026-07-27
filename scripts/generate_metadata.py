import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def date(year, month):
    return year + (month - 1) / 12


# Public model/checkpoint releases; Hugging Face creation dates are wrong for
# migrated repositories such as GPT-2.
RELEASE_DATES = {
    "openai-community/gpt2": date(2019, 2),
    "openai-community/gpt2-medium": date(2019, 5),
    "openai-community/gpt2-large": date(2019, 8),
    "openai-community/gpt2-xl": date(2019, 11),
    **{f"facebook/opt-{size}": date(2022, 5) for size in ["125m", "1.3b", "2.7b", "6.7b", "13b"]},
    **{f"EleutherAI/pythia-{size}-deduped": date(2023, 4) for size in ["410m", "1b", "1.4b", "6.9b", "12b"]},
    "allenai/OLMo-2-0425-1B": date(2025, 4),
    "allenai/OLMo-2-1124-7B": date(2024, 11),
    "allenai/OLMo-2-1124-13B": date(2024, 11),
    "meta-llama/Llama-3.2-1B": date(2024, 9),
    "meta-llama/Llama-3.2-3B": date(2024, 9),
    "meta-llama/Llama-3.1-8B": date(2024, 7),
    **{f"mistralai/Ministral-3-{size}B-Base-2512": date(2025, 12) for size in [3, 8, 14]},
    **{f"Qwen/Qwen3-{size}B-Base": date(2025, 4) for size in ["0.6", "1.7", "4", "8", "14"]},
    **{f"state-spaces/mamba-{size}-hf": date(2023, 12) for size in ["130m", "370m", "790m", "1.4b", "2.8b"]},
    **{f"AntonV/mamba2-{size}-hf": date(2024, 5) for size in ["130m", "370m", "780m", "1.3b", "2.7b"]},
    **{
        f"fla-hub/{model}": date(2025, 3)
        for model in ["rwkv7-191M-world", "rwkv7-0.4B-world", "rwkv7-1.5B-world", "rwkv7-2.9B-world", "rwkv7-7.2B-g0a"]
    },
}

# family, tokenizer, positional encoding, normalization, language focus
DESIGN = {
    "gpt2": ("GPT-2", "Byte-level BPE", "Learned absolute", "LayerNorm", "English-centric"),
    "opt": ("OPT", "Byte-level BPE", "Learned absolute", "LayerNorm", "English-centric"),
    "gpt_neox": ("Pythia", "Byte-level BPE", "RoPE", "LayerNorm", "English-centric"),
    "olmo2": ("OLMo-2", "Byte-level BPE", "RoPE", "RMSNorm", "English-centric"),
    "llama": ("Llama-3", "Byte-level BPE", "RoPE", "RMSNorm", "Multilingual"),
    "ministral3": ("Ministral-3", "Byte-level BPE", "RoPE", "RMSNorm", "Multilingual"),
    "qwen3": ("Qwen3", "Byte-level BPE", "RoPE", "RMSNorm", "Multilingual"),
    "mamba": ("Mamba", "Byte-level BPE", np.nan, "RMSNorm", "English-centric"),
    "mamba2": ("Mamba-2", "Byte-level BPE", np.nan, "RMSNorm", "English-centric"),
    "rwkv7": ("RWKV-7", "Trie", np.nan, "LayerNorm", "Multilingual"),
}


def get_first_attr(obj, *names):
    for name in names:
        value = obj.get(name)
        if value is not None:
            return value
    raise AttributeError(f"Could not find any of {names} in config")


def get_model_metadata(model_id, architecture, n_params_B, n_tokens_B):
    with open(hf_hub_download(model_id, "config.json")) as f:
        config = json.load(f)
    config = config.get("text_config", config)
    model_type = config["model_type"]
    family, tokenizer, positional, normalization, language_focus = DESIGN[model_type]
    depth = get_first_attr(config, "num_hidden_layers", "n_layer", "num_layers")
    width = get_first_attr(config, "hidden_size", "n_embd", "d_model")
    heads = config.get("num_attention_heads", config.get("n_head"))
    kv_heads = config.get("num_key_value_heads", heads)
    attention = np.nan if architecture != "transformer" else "Grouped-query" if kv_heads < heads else "Multi-head"
    activation = get_first_attr(config, "hidden_act", "activation_function").lower()
    activation = {"gelu_new": "GELU", "gelu": "GELU", "relu": "ReLU", "silu": "SiLU", "sqrelu": "SqReLU"}[activation]
    return {
        "model_name": model_id,
        "Family": family,
        "Architecture": architecture,
        "Num. Parameters": np.log10(n_params_B),
        "Release Date": RELEASE_DATES[model_id],
        "Depth": depth,
        "Width": np.log2(width),
        "Depth / Width": depth / width,
        "Training Tokens": np.log10(n_tokens_B),
        "Vocabulary Size": np.log2(config["vocab_size"]),
        "Tokenizer Type": tokenizer,
        "Language Focus": language_focus,
        "Positional Encoding": positional,
        "Attention Type": attention,
        "Normalization": normalization,
        "Activation": activation,
        "Tied Embeddings": str(config.get("tie_word_embeddings", model_type in {"gpt2", "opt"})),
    }


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

    df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "model_metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    generate_metadata()
