import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib.parallel import Parallel, delayed
from loguru import logger
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd()))

from src.reduce_dimensions import ReduceDimensions

path = Path("experiments/all")
assert (
    path.exists()
), f"{path} does not exist, the script should be run from the root directory"

logger.remove(0)
logger.add(sys.stderr, level="WARNING")
logger.add(path / ".logs" / "main.log", level="WARNING")


def process(model_name, layer, dataset, method):
    try:
        logs_path = (
            path
            / ".logs"
            / method
            / dataset
            / model_name.split("/")[-1]
            / f"layer_{layer}.log"
        )
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logger_id = logger.add(logs_path, backtrace=True, level="INFO")
        cfg = f"""
        dataset:
            csv_path: datasets/{dataset}.csv
        representations:
            level: {"word" if "word" in dataset else "sentence"}
            model_name: {model_name}
            layer: {layer}
        method: {method}
        """
        cfg = yaml.safe_load(cfg)
        rd = ReduceDimensions(**cfg)
        df = rd.transform()
        df["model_name"] = model_name
        df["layer"] = layer
        df["dataset"] = dataset
        df["method"] = method
        logger.remove(logger_id)

        return df, model_name, layer, dataset, method
    except KeyboardInterrupt:
        raise
    except Exception:
        exception = sys.exc_info()
        logger.opt(exception=exception).error(
            f"Error computing feature importance for {model_name} - {layer} - {dataset} - {method}"
        )


models = [
    ("gpt2", 13),
    # ("meta-llama/Llama-3.1-8B", 33),
    # ("mistralai/Mistral-7B-v0.3", 33),
    ("bert-base-uncased", 13),
    # ("microsoft/deberta-xlarge-mnli", 49),
    # ("McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp", 33),
]

datasets = [
    "svo_word_level",
    # "short_sentence",
    # "relative_clause",
    # "long_range_agreement",
]

methods = ["pca", "mds"]


# First compute representations, without multi processing
with tqdm(total=len(models) * len(datasets)) as pbar:
    for model_name, _ in models:
        for dataset in datasets:
            pbar.set_postfix_str(
                f"Processing {model_name} - 1 - {dataset} - {methods[0]}"
            )
            process(model_name, 1, dataset, methods[0])
            pbar.update(1)

# Then compute projections with multi processing for all layers
params = [
    (model_name, layer, dataset, method)
    for dataset in datasets
    for model_name, n_layers in models
    for layer in range(1, n_layers)
    for method in methods
]

# `require="sharedmem"` is a fix to make joblib Parallel work with loggers
for _, model_name, layer, dataset, method in (
    pbar := tqdm(
        Parallel(n_jobs=8, return_as="generator", require="sharedmem")(
            delayed(process)(model_name, layer, dataset, method)
            for model_name, layer, dataset, method in params
        ),
        total=len(params),
        desc="Projecting representations",
    )
):
    pbar.set_postfix_str(f"Processed {model_name} - {layer} - {dataset}")

# Then retrieve the results, aggregate and save
all_dfs = []
for df, _, _, _, _ in (
    pbar := tqdm(
        Parallel(
            n_jobs=8, return_as="generator", backend="threading", require="sharedmem"
        )(
            delayed(process)(model_name, layer, dataset, method)
            for model_name, layer, dataset, method in params
        ),
        total=len(params),
        desc="Retrieving results",
    )
):
    all_dfs.append(df)
all_dfs = pd.concat(all_dfs)
all_dfs.to_csv(path / "projections.csv.gz", index=False)
