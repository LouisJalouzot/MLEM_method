import json
import sys
from pathlib import Path

import yaml
from loguru import logger
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd()))

from src.feature_importance import FeatureImportance

path = Path("experiments/all_high_level")
assert (
    path.exists()
), f"{path} does not exist, the script should be run from the root directory"

for model_name, n_layers in [
    ("gpt2", 33),
    ("bert-base-uncased", 33),
    ("meta-llama/Llama-3.1-8B", 33),
    # ("mistralai/Mistral-7B-v0.3", 33),
    ("bchoiced/CHAIN19", 33),
    ("Linq-AI-Research/Linq-Embed-Mistral", 33),
    ("intfloat/5-mistral-7b-instruct", 33),
    ("microsoft/deberta-xlarge-mnli", 33),
]:
    for layer in tqdm(range(n_layers), desc=f"Model: {model_name}"):
        for dataset in [
            # "short_sentence",
            # "svo_word_level",
            "relative_clause",
            "long_range_agreement",
        ]:
            logs_path = path / "logs" / dataset / model_name / f"layer_{layer}.log"
            logs_path.parent.mkdir(parents=True, exist_ok=True)
            logger_id = logger.add(logs_path, level="DEBUG")
            cfg = f"""
            dataset:
                csv_path: datasets/{dataset}.csv
            trainer:
                representations:
                    level: {"word" if "word" in dataset else "sentence"}
                    model_name: {model_name}
                    layer: {layer}
                device: cpu
            """
            cfg = yaml.safe_load(cfg)
            fi = FeatureImportance(**cfg)
            with open(logs_path.parent / "config.json", "w") as f:
                json.dump(fi.model_dump(mode="json"), f, indent=4)
            fi.compute()
            logger.remove(logger_id)
            break
        break
