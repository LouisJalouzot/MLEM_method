import argparse
import importlib
import math
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from unflatten import unflatten

from mlem.utils import infra_cpu, infra_gpu


def yield_grid_search(grid_config):
    if not grid_config:
        yield {}
        return

    keys = grid_config.keys()
    values = grid_config.values()
    for v in product(*values):
        flat_config = dict(zip(keys, v))
        yield flat_config, unflatten(flat_config)


def run_grid_search(base_class, grid_search, method=None):
    flat_configs = []
    n_configs = math.prod(len(v) for v in grid_search.values())
    with base_class.infra.job_array(max_workers=n_configs) as array:
        with tqdm(total=n_configs, desc="Creating tasks") as pbar:
            for flat_config, task in Parallel(
                n_jobs=-2, return_as="generator", prefer="threads"
            )(
                delayed(
                    lambda flat_config, config: (
                        flat_config,
                        base_class.infra.clone_obj(config),
                    )
                )(flat_config, config)
                for flat_config, config in yield_grid_search(grid_search)
            ):
                flat_configs.append(flat_config)
                array.append(task)
                pbar.update(1)

    results = []
    for task in tqdm(array, desc="Waiting for completion and fetching results"):
        # Wait for computation to finish and retrieve result
        result = task.infra.job().result()
        if method is not None:
            result = getattr(task, method)()
        results.append(result)

    return flat_configs, results


def main(config: dict = {}):
    target = config.get("target", "src.feature_importance.FeatureImportance")

    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    TargetClass = getattr(module, class_name)
    base_config = config.get("base_config", {})
    method = config.get("method", None)

    if "grid_search_prepare" in config:
        base_config.update({"infra": infra_gpu})
        base_class = TargetClass(**base_config)
        print("Running grid search prepare on GPU")
        run_grid_search(
            base_class,
            config["grid_search_prepare"],
            method=method,
        )

    base_config.update({"infra": infra_cpu})
    base_class = TargetClass(**base_config)
    print("Running grid search on CPU")
    flat_configs, results = run_grid_search(
        base_class,
        config["grid_search"],
        method=method,
    )

    all_dfs = []
    for flat_config, dfs in zip(flat_configs, results):
        if not isinstance(dfs, (list, tuple)):
            dfs = [dfs]
        dfs = [df for df in dfs if isinstance(df, pd.DataFrame)]
        for df in dfs:
            for k, v in flat_config.items():
                try:
                    df[k] = v
                except Exception as e:
                    print(f"Error adding config {k}: {v} to DataFrame: {e}")
                    raise e
        all_dfs.append(dfs)

    for dfs in zip(*all_dfs):
        yield pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="*", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        for config_file in args.config:
            config_file = Path(config_file)
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            for i, df in enumerate(main(config)):
                df.to_parquet(config_file.parent / f"{i}.parquet")
    else:
        main()
