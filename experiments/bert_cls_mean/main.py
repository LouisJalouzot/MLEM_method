import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd()))

from src.feature_importance import FeatureImportance

path = Path("experiments/bert_cls_mean")
assert (
    path.exists()
), f"{path} does not exist, the script should be run from the root directory"

logger.remove(0)
logger.add(sys.stderr, level="WARNING")
logger.add(path / ".logs" / "main.log", level="INFO")

datasets = [
    "short_sentence",
    "relative_clause",
    "long_range_agreement",
]

all_importances = []
all_spearman = []
with tqdm(total=len(datasets) * 13 * 2) as pbar:
    for dataset in datasets:
        for token_aggregation in ["mean", "first"]:
            for layer in range(13):
                pbar.set_postfix_str(f"Processing {dataset} - {token_aggregation}")
                logger.warning(f"Processing {dataset} - {token_aggregation} - {layer}")
                cfg = f"""
                dataset:
                    csv_path: datasets/{dataset}.csv
                trainer:
                    representations:
                        model_name: bert-base-uncased
                        token_aggregation: {token_aggregation}
                        layer: {layer}
                """
                cfg = yaml.safe_load(cfg)
                fi = FeatureImportance(**cfg)
                importances, spearman = fi.compute()
                for e in [importances, spearman]:
                    e["dataset"] = dataset
                    e["token_aggregation"] = token_aggregation
                    e["layer"] = layer
                all_importances.append(importances)
                all_spearman.append(spearman)
                pbar.update(1)

all_importances = pd.concat(all_importances, ignore_index=True)
all_spearman = pd.concat(all_spearman, ignore_index=True)
all_importances.to_csv(path / "fi.csv.gz", index=False)
all_spearman.to_csv(path / "spearman.csv.gz", index=False)
