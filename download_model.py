import argparse

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_names",
    type=str,
    nargs="*",
    default=["answerdotai/ModernBERT-base"],
)
args = parser.parse_args()

for model_name in args.model_names:
    print("Downloading", model_name)
    snapshot_download(
        repo_id=model_name,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors_index.json", "*.md", "*.txt", "*.jsonl"],
    )
