import argparse
import importlib
import math
import sys
from contextlib import nullcontext
from functools import reduce
from itertools import batched, product
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import yaml
from exca import TaskInfra
from joblib import Parallel, delayed
from loguru import logger
from pydantic import BaseModel
from tqdm.auto import tqdm
from unflatten import unflatten


class TaskBatch(BaseModel):
    tasks: list[Any]
    uids: list[str]
    infra_name: str
    infra: TaskInfra = TaskInfra(mode="retry")
    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("tasks",)

    @infra.apply
    def compute(self):
        for task in self.tasks:
            infra = getattr(task, self.infra_name)
            local = infra.clone_obj(**{self.infra_name: {"cluster": None}})
            # Each task writes its original cache before the next task starts.
            getattr(local, self.infra_name).job().result()


def yield_grid_search(grid_config, grid_search_zip=None):
    keys = grid_config.keys()
    product_values = product(*grid_config.values()) if grid_config else [()]
    zipped = [{}]
    if grid_search_zip:
        lengths = {len(values) for values in grid_search_zip.values()}
        if len(lengths) != 1:
            raise ValueError("grid_search_zip values must have equal lengths")
        zipped = [dict(zip(grid_search_zip, values)) for values in zip(*grid_search_zip.values())]
    for values in product_values:
        base = dict(zip(keys, values))
        for point in zipped:
            flat_config = base | point
            yield flat_config, unflatten(flat_config)


def run_grid_search(
    base_class,
    grid_search,
    infra_path,
    fetch_results=True,
    max_workers=None,
    sequential=False,
    n_jobs=-2,
    grid_search_zip=None,
    tasks_per_job=1,
):
    """Run grid search with job array support.

    Args:
        base_class: The base pydantic class with infra.
        grid_search: Dict of parameter paths to lists of values.
        infra_path: Dotted path to the infra to use (e.g., 'trainer.representations.infra').
        fetch_results: If True, collect and return results. If False, just wait for completion.
        max_workers: Maximum number of cluster workers. Defaults to n_configs if not specified.
        sequential: Run each task locally, cancelling its pending cluster job first.
        n_jobs: Joblib workers used to construct tasks.
        tasks_per_job: Configurations run sequentially per allocation; ignored with sequential=True.
    """
    if not isinstance(tasks_per_job, int) or isinstance(tasks_per_job, bool) or tasks_per_job < 1:
        raise ValueError("tasks_per_job must be a positive integer")
    packed = not sequential and tasks_per_job > 1
    flat_configs = []
    n_configs = math.prod(len(v) for v in grid_search.values())
    if grid_search_zip:
        n_configs *= len(next(iter(grid_search_zip.values())))

    # Resolve which infra to use for cloning and job array
    infra_path_split = infra_path.split(".")
    base_infra = reduce(getattr, infra_path_split, base_class)

    # Create tasks and optionally submit them to a job array
    logger.info(f"Creating {n_configs} tasks for {base_class.__class__.__name__}.{infra_path}")
    logger.trace(f"Infra config: {base_infra.model_dump_json(indent=2)}")
    if sequential or packed:
        context = nullcontext([])
    else:
        context = base_infra.job_array(max_workers=max_workers or n_configs, allow_repeated_tasks=True)

    with context as array:
        with tqdm(total=n_configs, desc="Creating tasks") as pbar:
            for flat_config, task in Parallel(n_jobs=n_jobs, return_as="generator", prefer="threads")(
                delayed(lambda flat_config, config: (flat_config, base_infra.clone_obj(config)))(flat_config, config)
                for flat_config, config in yield_grid_search(grid_search, grid_search_zip)
            ):
                flat_configs.append(flat_config)
                array.append(task)
                pbar.update(1)
        if not sequential and not packed:
            logger.info("Submitting tasks to job array")

    if packed:
        # Import by module name so Submitit can reload batches launched via the CLI.
        from main import TaskBatch

        infra_name = infra_path_split[-1]
        batch_config = dict(infra_name=infra_name, infra=base_infra.model_dump())
        template = TaskBatch(tasks=[], uids=[], **batch_config)
        with template.infra.job_array(
            max_workers=max_workers or math.ceil(n_configs / tasks_per_job),
            allow_empty=True,
            allow_repeated_tasks=True,
        ) as batches:
            for tasks in batched(array, tasks_per_job):
                infras = [getattr(task, infra_name) for task in tasks]
                if base_infra.mode != "force" and all(infra.status() == "completed" for infra in infras):
                    continue
                batch = TaskBatch(tasks=list(tasks), uids=[infra.uid() for infra in infras], **batch_config)
                # A completed batch receipt may outlive an individually cleared task cache.
                if batch.infra.status() == "completed":
                    batch.infra.clear_job()
                batches.append(batch)
        for batch in tqdm(batches, desc="Waiting for packed jobs"):
            batch.infra.job().result()

    # Wait for completion and collect results if necessary
    results = []
    has_error = False
    desc = "Waiting for completion"
    if fetch_results:
        desc += " and fetching results"
    for idx, task in enumerate(tqdm(array, desc=desc)):
        task_infra = getattr(task, infra_path_split[-1])
        try:
            if packed:
                task = task_infra.clone_obj(**{infra_path_split[-1]: {"mode": "cached"}})
                task_infra = getattr(task, infra_path_split[-1])
            if sequential:
                try:
                    cached = task_infra.status() == "completed"
                except FileNotFoundError:
                    cached = False
                if not cached:
                    task_infra.clear_job()
                task = task_infra.clone_obj(**{infra_path_split[-1]: {"cluster": None}})
                task_infra = getattr(task, infra_path_split[-1])
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


def main(config: dict | None = None, n_jobs=-2):
    config = config or {}
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
            sequential=config.get("sequential", False),
            tasks_per_job=config.get("tasks_per_job", 1),
            n_jobs=n_jobs,
        )

    logger.info("Running grid search")
    flat_configs, results = run_grid_search(
        base_class,
        config["grid_search"],
        infra_path=infra_path,
        max_workers=config.get("max_workers"),
        sequential=config.get("sequential", False),
        tasks_per_job=config.get("tasks_per_job", 1),
        n_jobs=n_jobs,
        grid_search_zip=config.get("grid_search_zip"),
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
                    raise
        all_dfs.append(dfs)

    for dfs in zip(*all_dfs):
        yield pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="*", type=str, default=None)
    parser.add_argument("--sequential", action="store_true", help="Run jobs one after another on this node.")
    parser.add_argument(
        "--tasks-per-job",
        type=int,
        default=None,
        help="Run N configurations sequentially per allocation (default: 1; ignored with --sequential).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-2,
        help="Joblib workers used to construct tasks (default: -2; use 1 if the backend is not thread-safe).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()
    if args.tasks_per_job is not None and args.tasks_per_job < 1:
        parser.error("--tasks-per-job must be positive")

    # Configure logging level
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if args.config:
        for config_file in args.config:
            config_file = Path(config_file)
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            config["sequential"] = args.sequential
            if args.tasks_per_job is not None:
                config["tasks_per_job"] = args.tasks_per_job
            for i, df in enumerate(main(config, n_jobs=args.n_jobs)):
                df.to_parquet(config_file.parent / f"{i}.parquet")
    else:
        main(n_jobs=args.n_jobs)
