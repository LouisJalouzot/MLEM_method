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


def yield_grid_search(grid_config):
    if not grid_config:
        yield {}
        return

    keys = grid_config.keys()
    values = grid_config.values()
    for v in product(*values):
        flat_config = dict(zip(keys, v))
        yield flat_config, unflatten(flat_config)


def run_grid_search(base_class, grid_search, infra_path, fetch_results=True, max_workers=None):
    """Run grid search with job array support.

    Args:
        base_class: The base pydantic class with infra.
        grid_search: Dict of parameter paths to lists of values.
        infra_path: Dotted path to the infra to use (e.g., 'trainer.representations.infra').
        fetch_results: If True, collect and return results. If False, just wait for completion.
        max_workers: Maximum number of workers. Defaults to n_configs if not specified.
    """
    flat_configs = []
    n_configs = math.prod(len(v) for v in grid_search.values())

    # Resolve which infra to use for cloning and job array
    infra_path_split = infra_path.split(".")
    base_infra = reduce(getattr, infra_path_split, base_class)

    # Create tasks and submit to job array
    logger.info(f"Submitting {n_configs} tasks to {base_class.__class__.__name__}.{infra_path}")
    logger.trace(f"Infra config: {base_infra.model_dump_json(indent=2)}")
    with base_infra.job_array(max_workers=max_workers or n_configs) as array:
        with tqdm(total=n_configs, desc="Creating tasks") as pbar:
            for flat_config, task in Parallel(n_jobs=-2, return_as="generator", prefer="threads")(
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
        logger.info("Submitting tasks to job array")
    logger.info("All tasks submitted to job array")

    # Wait for completion and collect results if necessary
    results = []
    has_error = False
    desc = "Waiting for completion"
    if fetch_results:
        desc += " and fetching results"
    for idx, task in enumerate(tqdm(array, desc=desc)):
        task_infra = getattr(task, infra_path_split[-1])
        try:
            job = task_infra.job()
            # Wait for completion and log exception if any
            exc = job.exception()
            if exc is not None:
                has_error = True
                logger.error(f"Config: {flat_configs[idx]} | Cache: {Path(task_infra.uid_folder()).resolve()}")
                logger.error(exc)
            if fetch_results:
                results.append(job.result())
        except KeyboardInterrupt:
            logger.error(
                f"Keyboard interrupt | Config: {flat_configs[idx]} | Cache: {Path(task_infra.uid_folder()).resolve()}"
            )
            raise

    if has_error:
        raise RuntimeError("Grid search encountered errors")

    return flat_configs, results


def main(config: dict = {}):
    target = config.get("target", "mlem_method.FeatureImportance")
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    TargetClass = getattr(module, class_name)
    base_class = TargetClass(**config.get("base_config", {}))
    infra_path = config.get("infra", "infra")
    infra_prepare = config.get("infra_prepare", "infra")

    if "grid_search_prepare" in config:
        logger.info("Running grid search prepare")
        run_grid_search(
            base_class,
            config["grid_search_prepare"],
            infra_path=infra_prepare,
            fetch_results=False,
            max_workers=config.get("max_workers"),
        )

    logger.info("Running grid search")
    flat_configs, results = run_grid_search(
        base_class,
        config["grid_search"],
        infra_path=infra_path,
        max_workers=config.get("max_workers"),
    )

    all_dfs = []
    for flat_config, dfs in zip(flat_configs, results):
        if not isinstance(dfs, (list, tuple)):
            dfs = [dfs]
        dfs = [df for df in dfs if isinstance(df, pd.DataFrame)]
        for df in dfs:
            for k, v in flat_config.items():
                try:
                    df[k] = str(v)
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
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
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
