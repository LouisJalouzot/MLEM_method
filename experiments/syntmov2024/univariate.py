# %%
import os
from pathlib import Path

from mlem.syntmov2024_dataset import SyntMov2024Dataset
from mlem.univariate import UnivariateAnalysis

os.chdir(Path(__file__).parent.parent.parent)

# %%
uni = UnivariateAnalysis(
    feature_importance=dict(
        dataset=SyntMov2024Dataset(),
        trainer=dict(
            representations=dict(level="syntmov2024", map_infra=dict(cluster=None)),
            dataloader_builder=dict(cv=0.2),
        ),
    ),
    map_infra=dict(cluster=None),
)
all_i, all_s, all_w = uni.run()
