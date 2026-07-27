# %%
import os
from pathlib import Path

from mlem_method.feature_importance import FeatureImportance
from mlem_method.syntmov2024_dataset import SyntMov2024Dataset

os.chdir(Path(__file__).parent.parent.parent)

# %%
fi = FeatureImportance(
    dataset=SyntMov2024Dataset(),
    trainer=dict(representations=dict(level="syntmov2024")),
)
i, s, w = fi.compute()
