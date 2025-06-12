import argparse
import sys

import pandas as pd
import yaml
from loguru import logger
from tqdm.auto import tqdm

from src.feature_importance import FeatureImportance
from src.utils import infra_cpu, infra_gpu

logger.remove()
logger.add(sys.stderr, level="WARNING")


def get_cfg(method, max_epochs):
    cfg = f"""
    dataset:
        path: datasets/relative_clause.csv
    trainer:
        dataloader_builder:
            cv: 5
        model_builder:
            param: {method}
        max_epochs: {max_epochs}
        representations:
            model_name: bert-base-uncased
            layer: 7
        eps: 1e-5
    """
    cfg = yaml.safe_load(cfg)

    return cfg


methods = [
    ("triu", "FR-RSA"),
    ("cholesky", "MLEM cholesky"),
    ("exp", "MLEM exp"),
]
max_epochs = [1, 5, 10, 20, 30, 50, 100, 200, 300, 500, 750, 1000]


def launch():
    # Launch the first task to ensure LLM embeddings are cached
    cfg = get_cfg(methods[0][0], max_epochs[0])
    cfg["infra"] = infra_gpu
    fi = FeatureImportance(**cfg)
    fi.compute()

    cfg["infra"] = infra_cpu
    fi = FeatureImportance(**cfg)
    with tqdm(total=len(methods) * len(max_epochs), desc="Creating tasks") as pbar:
        with fi.infra.job_array() as array:
            for param, _ in methods:
                for k in max_epochs:
                    task_to_compute = fi.infra.clone_obj(
                        {"trainer": {"max_epochs": k, "model_builder": {"param": param}}}
                    )
                    array.append(task_to_compute)
                    pbar.update(1)


def fetch():
    cfg = get_cfg(methods[0][0], max_epochs[0])
    results = []
    for param, method in methods:
        for k in max_epochs:
            fi = FeatureImportance(**cfg)
            task_to_compute = fi.infra.clone_obj(
                {"trainer": {"max_epochs": k, "model_builder": {"param": param}}}
            )
            results.append([param, k, task_to_compute])

    importances, spearman, weights = [], [], []
    with tqdm(total=len(methods) * len(max_epochs), desc="Creating tasks") as pbar:
        for param, method in methods:
            for k in max_epochs:
                cfg = get_cfg(param, k)
                fi = FeatureImportance(**cfg)
                i, s, w = fi.compute()
                for e in [i, s, w]:
                    e["Method"] = method
                    e["max_epochs"] = k
                importances.append(i)
                spearman.append(s)
                weights.append(w)
                pbar.update(1)
    pd.concat(importances).to_parquet("experiments/fr_rsa/importances.parquet")
    pd.concat(spearman).to_parquet("experiments/fr_rsa/spearman.parquet")
    pd.concat(weights).to_parquet("experiments/fr_rsa/weights.parquet")


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
