# %%
import os
import sys
from pathlib import Path

from loguru import logger

from mlem import SyntMov2024Dataset, UnivariateAnalysis, map_infra_cpu

logger.remove()
logger.add(sys.stdout, level="WARNING")

os.chdir(Path(__file__).parent.parent.parent)

# %%
uni = UnivariateAnalysis(
    feature_importance=dict(
        dataset=SyntMov2024Dataset(bids_root="data/syntmov2024/fmriprep"),
        trainer=dict(
            representations=dict(level="syntmov2024", map_infra=dict(cluster=None)),
            dataloader_builder=dict(cv=0.2),
        ),
    ),
    map_infra=map_infra_cpu,
)
all_i, all_s, all_w = uni.run()
