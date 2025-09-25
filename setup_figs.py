import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.utils import shuffle
from tqdm.auto import tqdm

from src.estimate_correlations import EstimateCorrelations
from src.feature_importance import FeatureImportance
from src.reduce_dimensions import ReduceDimensions

sns.set_context("poster")
sns.set_style("ticks")

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["figure.titleweight"] = "bold"
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True

rename = {
    "sentence_CLAUSE": "Relative clause type",
    "sentence_RC_attached": "Attachment site",
    "subj_NUM": "Subj. num.",
    "subj_GEN": "Subj. gender",
    "subj_ZIPF": "Subj. freq.",
    "obj_NUM": "Obj. num.",
    "obj_GEN": "Obj. gender",
    "obj_ZIPF": "Obj. freq.",
    "embed_NUM": "Embed. num.",
    "embed_GEN": "Embed. gender",
    "embed_ZIPF": "Embed. freq.",
    "verb_ZIPF": "Verb freq.",
    "sg": "Singular",
    "pl": "Plural",
    "peripheral": "Peripheral",
    "center_embedding": "Center embedding",
    "subjwho": "Subj. relative",
    "objwho": "Obj. relative",
    "triu": "FR-RSA-I",
    "cholesky": "MLEM",
}


def _rename_label(label):
    new_label = str(label)
    for old, new in rename.items():
        new_label = new_label.replace(old, new)
    return new_label


def clean_names(df):
    df = df.rename(columns=_rename_label, index=_rename_label).replace(rename)
    if "Feature" in df:
        df["Feature"] = df["Feature"].apply(_rename_label)
    return df
