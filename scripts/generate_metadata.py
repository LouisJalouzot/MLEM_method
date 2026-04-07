from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import model_info
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from transformers import AutoConfig


def get_model_metadata(model_id, architecture, training_tokens_B):
    info = model_info(model_id)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Extract features
    n_params = getattr(info, "safetensors", {}).get("total", 0) if hasattr(info, "safetensors") else 0
    if n_params == 0:
        n_params = getattr(info, "params", {}).get("total", 0)

    log_n_params_B = np.log10(n_params / 1e9) if n_params > 0 else np.nan
    release_year = info.created_at.year if info.created_at else np.nan

    # Depth/Width/Vocab from config
    n_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", getattr(config, "num_layers")))
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", getattr(config, "d_model")))

    return [
        model_id,
        architecture,
        np.log2(log_n_params_B),
        int(release_year) if not np.isnan(release_year) else np.nan,
        np.log2(n_layers),
        np.log2(hidden_size),
        np.log2(n_layers) / np.log2(hidden_size),
        getattr(config, "vocab_size"),
        np.log10(training_tokens_B) if training_tokens_B else None,
    ]


def generate_metadata():
    # model_id, architecture, training_tokens_B
    models_to_fetch = [
        ("openai-community/gpt2", "transformer", None),
        ("openai-community/gpt2-medium", "transformer", None),
        ("openai-community/gpt2-large", "transformer", None),
        ("openai-community/gpt2-xl", "transformer", None),
        ("facebook/opt-125m", "transformer", 180),
        ("facebook/opt-1.3b", "transformer", 180),
        ("facebook/opt-2.7b", "transformer", 180),
        ("facebook/opt-6.7b", "transformer", 180),
        ("facebook/opt-13b", "transformer", 180),
        ("EleutherAI/pythia-410m-deduped", "transformer", 207),
        ("EleutherAI/pythia-1b-deduped", "transformer", 207),
        ("EleutherAI/pythia-1.4b-deduped", "transformer", 207),
        ("EleutherAI/pythia-6.9b-deduped", "transformer", 207),
        ("EleutherAI/pythia-12b-deduped", "transformer", 207),
        ("allenai/OLMo-2-0425-1B", "transformer", 4000),
        ("allenai/OLMo-2-1124-7B", "transformer", 4000),
        ("allenai/OLMo-2-1124-13B", "transformer", 5000),
        ("meta-llama/Llama-3.2-1B", "transformer", 9000),
        ("meta-llama/Llama-3.2-3B", "transformer", 9000),
        ("meta-llama/Llama-3.1-8B", "transformer", 15000),
        ("mistralai/Ministral-3-3B-Base-2512", "transformer", 3000),
        ("mistralai/Ministral-3-8B-Base-2512", "transformer", 3000),
        ("mistralai/Ministral-3-14B-Base-2512", "transformer", 3000),
        ("Qwen/Qwen3-0.6B-Base", "transformer", 36000),
        ("Qwen/Qwen3-1.7B-Base", "transformer", 36000),
        ("Qwen/Qwen3-4B-Base", "transformer", 36000),
        ("Qwen/Qwen3-8B-Base", "transformer", 36000),
        ("Qwen/Qwen3-14B-Base", "transformer", 36000),
        ("state-spaces/mamba-130m-hf", "mamba", 300),
        ("state-spaces/mamba-370m-hf", "mamba", 300),
        ("state-spaces/mamba-790m-hf", "mamba", 300),
        ("state-spaces/mamba-1.4b-hf", "mamba", 300),
        ("state-spaces/mamba-2.8b-hf", "mamba", 300),
        ("AntonV/mamba2-130m-hf", "mamba", 300),
        ("AntonV/mamba2-370m-hf", "mamba", 300),
        ("AntonV/mamba2-780m-hf", "mamba", 300),
        ("AntonV/mamba2-1.3b-hf", "mamba", 300),
        ("AntonV/mamba2-2.7b-hf", "mamba", 300),
        ("fla-hub/rwkv7-191M-world", "rwkv", 1600),
        ("fla-hub/rwkv7-0.4B-world", "rwkv", 3100),
        ("fla-hub/rwkv7-1.5B-world", "rwkv", 5600),
        ("fla-hub/rwkv7-2.9B-world", "rwkv", 5600),
        ("fla-hub/rwkv7-7.2B-g0a", "rwkv", 5600),
    ]

    results = tqdm(
        Parallel(n_jobs=-2, return_as="generator", backend="threading")(
            delayed(get_model_metadata)(m_id, arch, tokens) for m_id, arch, tokens in models_to_fetch
        ),
        total=len(models_to_fetch),
        desc="Fetching model metadata",
    )

    columns = [
        "model_name",
        "architecture",
        "log_n_params_B",
        "release_year",
        "log_depth",
        "log_width",
        "ratio",
        "vocab_size_k",
        "log_training_tokens_T",
    ]
    df = pd.DataFrame(results, columns=columns)
    output_path = Path(__file__).parent.parent / "model_metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    generate_metadata()
