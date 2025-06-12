import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.feature_importance import FeatureImportance
from src.utils import infra_cpu

logger.remove()
logger.add(sys.stderr, level="WARNING")


def get_cfg(method, max_epochs, noise_level):
    cfg = f"""
    dataset:
        path: datasets/simulated.csv
    trainer:
        dataloader_builder:
            cv: 5
        model_builder:
            param: {method}
        max_epochs: {max_epochs}
        representations:
            level: simulated
            noise_level: {noise_level}
        eps: 1e-5
    """
    cfg = yaml.safe_load(cfg)

    return cfg


methods = [
    ("diagonal", "FR-RSA"),
    ("triu", "FR-RSA + interactions"),
    ("cholesky", "MLEM cholesky"),
    ("exp", "MLEM exp"),
]
max_epochs = [1, 5, 10, 20, 30, 50, 100, 200, 300, 500, 750, 1000]
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]


def launch():
    cfg = get_cfg(methods[0][0], max_epochs[0], noise_levels[0])
    cfg["infra"] = infra_cpu
    fi = FeatureImportance(**cfg)
    with tqdm(
        total=len(methods) * len(max_epochs) * len(noise_levels), desc="Creating tasks"
    ) as pbar:
        with fi.infra.job_array() as array:
            for param, _ in methods:
                for k in max_epochs:
                    for noise_level in noise_levels:
                        task_to_compute = fi.infra.clone_obj(
                            {
                                "trainer": {
                                    "max_epochs": k,
                                    "model_builder": {"param": param},
                                    "representations": {"noise_level": noise_level},
                                }
                            }
                        )
                        array.append(task_to_compute)
                        pbar.update(1)


def fetch():
    importances, spearman, weights = [], [], []
    with tqdm(
        total=len(methods) * len(max_epochs) * len(noise_levels), desc="Fetching results"
    ) as pbar:
        for param, method in methods:
            for k in max_epochs:
                for noise_level in noise_levels:
                    cfg = get_cfg(param, k, noise_level)
                    cfg["infra"] = infra_cpu
                    fi = FeatureImportance(**cfg)
                    i, s, w = fi.compute()
                    for e in [i, s, w]:
                        e["Method"] = method
                        e["max_epochs"] = k
                        e["noise_level"] = noise_level
                    importances.append(i)
                    spearman.append(s)
                    weights.append(w)
                    pbar.update(1)
    pd.concat(importances).to_parquet("experiments/simulations/importances.parquet")
    pd.concat(spearman).to_parquet("experiments/simulations/spearman.parquet")
    pd.concat(weights).to_parquet("experiments/simulations/weights.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=["launch", "fetch"],
        default="launch",
        help="Action to perform: launch or fetch",
    )
    args = parser.parse_args()

    if args.action == "launch":
        launch()
    elif args.action == "fetch":
        fetch()
