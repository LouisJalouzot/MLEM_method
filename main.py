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

from src.utils import infra_cpu, infra_gpu


def yield_grid_search(grid_config):
    if not grid_config:
        yield {}

    keys = grid_config.keys()
    values = grid_config.values()
    for v in product(*values):
        config = dict(zip(keys, v))
        yield config, unflatten(config)


def main(config: dict = {}):
    target = config.get("target", "src.feature_importance.FeatureImportance")

    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    TargetClass = getattr(module, class_name)
    base_config = config.get("base_config", {})
    base_config.update({"infra": infra_gpu})

    grid_search = config.get("grid_search", {})
    if not grid_search:
        n_configs = 1
    else:
        n_configs = math.prod(len(v) for v in grid_search.values())
    grid_search = yield_grid_search(grid_search)

    flat_configs = []

    print("Running first config on GPU")
    flat_first_config, first_config = next(grid_search)
    flat_configs.append(flat_first_config)
    first_config.update(base_config)
    first_config.update({"infra": infra_gpu})
    results = [TargetClass(**first_config).infra.job().result()]

    def aux(flat_config, config):
        return flat_config, base_class.infra.clone_obj(config)

    base_config.update({"infra": infra_cpu})
    base_class = TargetClass(**base_config)
    with base_class.infra.job_array() as array:
        with tqdm(total=n_configs - 1, desc="Creating tasks") as pbar:
            for flat_config, task in Parallel(
                n_jobs=-2, return_as="generator", prefer="threads"
            )(delayed(aux)(flat_config, config) for flat_config, config in grid_search):
                flat_configs.append(flat_config)
                array.append(task)
                pbar.update(1)

    for task in tqdm(array, desc="Waiting for completion and fetching results"):
        results.append(task.infra.job().result())

    all_dfs = []
    for flat_config, dfs in zip(flat_configs, results):
        for df in dfs:
            for k, v in flat_config.items():
                df[k] = v
        all_dfs.append(dfs)

    for i, dfs in enumerate(zip(*all_dfs)):
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
