import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib.parallel import Parallel, delayed
from loguru import logger
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd()))

from src.baselines import DecodingBaseline, EncodingBaseline

path = Path("experiments/baselines")
assert (
    path.exists()
), f"{path} does not exist, the script should be run from the root directory"

logger.remove(0)
logger.add(sys.stderr, level="WARNING")
logger.add(path / ".logs" / "main.log", level="WARNING")


def process(model_name, layer, dataset):
    try:
        logs_path = (
            path / ".logs" / dataset / model_name.split("/")[-1] / f"layer_{layer}.log"
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
        """
        cfg = yaml.safe_load(cfg)
        encoding = EncodingBaseline(**cfg).compute()
        decoding = DecodingBaseline(**cfg).compute()
        for e in [encoding, decoding]:
            e["model_name"] = model_name
            e["layer"] = layer
            e["dataset"] = dataset
        logger.remove(logger_id)

        return encoding, decoding, model_name, layer, dataset
    except KeyboardInterrupt:
        raise
    except Exception:
        exception = sys.exc_info()
        logger.opt(exception=exception).error(
            f"Error processing {model_name} - {layer} - {dataset}"
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


# First compute representations, without multi processing
with tqdm(total=len(models) * len(datasets)) as pbar:
    for model_name, _ in models:
        for dataset in datasets:
            pbar.set_postfix_str(f"Processing {model_name} - 1 - {dataset}")
            process(model_name, 1, dataset)
            pbar.update(1)

# Then compute feature importance with multi processing for all layers
params = [
    (model_name, layer, dataset)
    for dataset in datasets
    for model_name, n_layers in models
    for layer in range(1, n_layers)
]

# `require="sharedmem"` is a fix to make joblib Parallel work with loggers
for _, _, model_name, layer, dataset in (
    pbar := tqdm(
        Parallel(n_jobs=1, return_as="generator", require="sharedmem")(
            delayed(process)(model_name, layer, dataset)
            for model_name, layer, dataset in params
        ),
        total=len(params),
        desc="Computing baselines",
    )
):
    pbar.set_postfix_str(f"Processed {model_name} - {layer} - {dataset}")

all_encoding = []
all_decoding = []
# `require="sharedmem"` is a fix to make joblib Parallel work with loggers
for encoding, decoding, model_name, layer, dataset in (
    pbar := tqdm(
        Parallel(
            n_jobs=8, return_as="generator", backend="threading", require="sharedmem"
        )(
            delayed(process)(model_name, layer, dataset)
            for model_name, layer, dataset in params
        ),
        total=len(params),
        desc="Retrieving results",
    )
):
    all_encoding.append(encoding)
    all_decoding.append(decoding)
    pbar.set_postfix_str(f"Processed {model_name} - {layer} - {dataset}")

all_encoding = pd.concat(all_encoding, ignore_index=True)
all_encoding.to_csv(path / "encoding.csv.gz", index=False)
all_decoding = pd.concat(all_decoding, ignore_index=True)
all_decoding.to_csv(path / "decoding.csv.gz", index=False)
