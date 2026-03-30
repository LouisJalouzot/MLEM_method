from pathlib import Path

import pandas as pd

from mlem import ReduceDimensions

script_dir = Path(__file__).parent

dfs = []
model_layers = [
    ("AntonV/mamba2-1.3b-hf", 21),
    ("openai-community/gpt2-medium", 11),
    ("fla-hub/rwkv7-191M-world", 6),
    ("EleutherAI/pythia-6.9b-deduped", 14),
    ("state-spaces/mamba-790m-hf", 21),
    ("Qwen/Qwen3-4B-Base", 16),
    ("facebook/opt-1.3b", 17),
    ("Qwen/Qwen2.5-7B", 20),
    ("fla-hub/rwkv7-191M-world", 9),
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
