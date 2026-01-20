import argparse
import importlib
import math
import sys
from functools import reduce
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from loguru import logger
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


def run_grid_search(base_class, grid_search, infra_path=None):
    """Run grid search with job array support.

    Args:
        base_class: The base pydantic class with infra.
        grid_search: Dict of parameter paths to lists of values.
        infra_path: Optional dotted path to the infra to use (e.g., 'trainer.representations.infra').
                    If None, uses base_class.infra.
    """
    flat_configs = []
    n_configs = math.prod(len(v) for v in grid_search.values())

    # Resolve which infra to use for cloning and job array
    if infra_path is None:
        infra_attr = "infra"
    else:
        infra_attr = infra_path.split(".")[-1]
    base_infra = (
        getattr(base_class, infra_attr)
        if infra_path is None
        else reduce(getattr, infra_path.split("."), base_class)
    )

    # Create tasks and submit to job array
    with base_infra.job_array(max_workers=n_configs) as array:
        with tqdm(total=n_configs, desc="Creating tasks") as pbar:
            for flat_config, task in Parallel(
                n_jobs=-2, return_as="generator", prefer="threads"
            )(
                delayed(
                    lambda flat_config, config: (
                        flat_config,
                        base_infra.clone_obj(config),
                    )
                )(flat_config, config)
                for flat_config, config in yield_grid_search(grid_search)
            ):
                flat_configs.append(flat_config)
                array.append(task)
                pbar.update(1)

    # Collect results
    results = []
    has_error = False

    for idx, task in enumerate(tqdm(array, desc="Fetching results")):
        try:
            task_infra = (
                reduce(getattr, infra_path.split("."), task)
                if infra_path
                else task.infra
            )
            result = task_infra.job().result()
            results.append(result)
        except Exception as e:
            has_error = True
            task_infra = (
                reduce(getattr, infra_path.split("."), task)
                if infra_path
                else task.infra
            )
            logger.error(
                f"Config: {flat_configs[idx]} | Cache: {Path(task_infra.uid_folder()).resolve()}"
            )
            logger.exception(e)
            results.append(None)

    if has_error:
        raise RuntimeError("Grid search encountered errors")

    return flat_configs, results


def main(config: dict = {}):
    target = config.get("target", "mlem.FeatureImportance")

    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    TargetClass = getattr(module, class_name)
    base_config = config.get("base_config", {})
    infra_path = config.get("infra", None)
    infra_prepare = config.get("infra_prepare", None)

    if "grid_search_prepare" in config:
        base_config.update({"infra": infra_gpu})
        base_class = TargetClass(**base_config)
        logger.info(
            f"Running grid search prepare on GPU (infra: {infra_prepare or 'default'})"
        )
        run_grid_search(
            base_class,
            config["grid_search_prepare"],
            infra_path=infra_prepare,
        )

    base_config.update({"infra": infra_cpu})
    base_class = TargetClass(**base_config)
    logger.info(f"Running grid search on CPU (infra: {infra_path or 'default'})")
    flat_configs, results = run_grid_search(
        base_class,
        config["grid_search"],
        infra_path=infra_path,
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: ERROR)",
    )
    args = parser.parse_args()

    # Configure logging level
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if args.config:
        for config_file in args.config:
            config_file = Path(config_file)
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            for i, df in enumerate(main(config)):
                df.to_parquet(config_file.parent / f"{i}.parquet")
    else:
        main()
