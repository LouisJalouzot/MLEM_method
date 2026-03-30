from pathlib import Path

import pandas as pd

from mlem import ReduceDimensions

script_dir = Path(__file__).parent

dfs = []
model_layers = [
    ("openai-community/gpt2-medium", 11),
    ("openai-community/gpt2-large", 2),
    ("facebook/opt-125m", 4),
    ("facebook/opt-1.3b", 17),
    ("Qwen/Qwen3-0.6B-Base", 9),
    ("Qwen/Qwen3-1.7B-Base", 9),
    ("Qwen/Qwen3-4B-Base", 11),
    ("Qwen/Qwen3-4B-Base", 16),
    ("Qwen/Qwen3-8B-Base", 11),
    ("EleutherAI/pythia-1.4b-deduped", 8),
    ("EleutherAI/pythia-1.4b-deduped", 11),
    ("EleutherAI/pythia-6.9b-deduped", 10),
    ("EleutherAI/pythia-6.9b-deduped", 14),
    ("state-spaces/mamba-790m-hf", 11),
    ("state-spaces/mamba-790m-hf", 21),
    ("AntonV/mamba2-780m-hf", 14),
    ("AntonV/mamba2-1.3b-hf", 21),
    ("fla-hub/rwkv7-191M-world", 6),
    ("fla-hub/rwkv7-191M-world", 9),
    ("fla-hub/rwkv7-1.5B-world", 8),
    ("fla-hub/rwkv7-7.2B-g0a", 10),
]
for method in ["pca", "mds"]:
    rd = ReduceDimensions(
        dataset=dict(path="datasets/relative_clause.csv"),
        representations=dict(token_aggregation="last"),
        method=method,
    )
    df = rd.transform_multiple(model_layers)
    df["method"] = method
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.to_parquet(script_dir / "proj.parquet")
