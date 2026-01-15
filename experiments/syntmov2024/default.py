# %%
import os
from pathlib import Path

from mlem.feature_importance import FeatureImportance
from mlem.syntmov2024_dataset import SyntMov2024Dataset
from mlem.syntmov2024_representations import SyntMov2024Representations

os.chdir(Path(__file__).parent.parent.parent)

ds = SyntMov2024Dataset()
reps = SyntMov2024Representations(dataset=ds, map_infra=dict(max_jobs=3))

df = ds.df
df_features = ds.df_features
X = ds.encode()
Y = reps()

# %%
fi = FeatureImportance(dataset=ds, trainer=dict(dataset=ds, representations=reps))
i, s, w = fi.compute()
