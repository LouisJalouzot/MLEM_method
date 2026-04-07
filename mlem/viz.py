import typing as tp
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from dtw import dtw
from huggingface_hub import model_info
from joblib import Parallel, delayed
from plotly.colors import sample_colorscale
from scipy.ndimage import gaussian_filter
from scipy.stats import weightedtau
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
plt.rcParams["font.family"] = "serif"
pio.templates.default = "simple_white"
markers = ["o", "s", "^", "v", "D", "p"]
palettes = [
    "Reds",
    "Greens",
    "Blues",
    "Oranges",
    "Purples",
    "Greys",
]

metadata = pd.DataFrame(
    [
        ["openai-community/gpt2", "T", 0.117, 2019, 12, 768, 0.016, 50257, pd.NA],
        ["openai-community/gpt2-medium", "T", 0.345, 2019, 24, 1024, 0.023, 50, pd.NA],
        ["openai-community/gpt2-large", "T", 0.762, 2019, 36, 1280, 0.028, 50, pd.NA],
        ["openai-community/gpt2-xl", "T", 1.5, 2019, 48, 1600, 0.03, 50, pd.NA],
        ["EleutherAI/pythia-410m-deduped", "T", 0.41, 2023, 24, 1024, 0.023, 50, 207],
        ["EleutherAI/pythia-1b-deduped", "T", 1.0, 2023, 16, 2048, 0.0078, 50, 207],
        ["EleutherAI/pythia-1.4b-deduped", "T", 1.4, 2023, 24, 2048, 0.012, 50, 207],
        ["EleutherAI/pythia-6.9b-deduped", "T", 6.9, 2023, 32, 4096, 0.0078, 50, 207],
        ["EleutherAI/pythia-12b-deduped", "T", 12.0, 2023, 36, 5120, 0.007, 50, 207],
        ["facebook/opt-125m", "T", 0.125, 2022, 12, 768, 0.016, 50, 180],
        ["facebook/opt-1.3b", "T", 1.3, 2022, 24, 2048, 0.012, 50, 180],
        ["facebook/opt-2.7b", "T", 2.7, 2022, 32, 2560, 0.013, 50, 180],
        ["facebook/opt-6.7b", "T", 6.7, 2022, 32, 4096, 0.0078, 50, 180],
        ["facebook/opt-13b", "T", 13.0, 2022, 40, 5120, 0.0078, 50, 180],
        ["allenai/OLMo-2-0425-1B", "T", 1.0, 2025, 16, 2048, 0.0078, 100, 4000],
        ["allenai/OLMo-2-1124-7B", "T", 7.0, 2025, 32, 4096, 0.0078, 100, 4000],
        ["allenai/OLMo-2-1124-13B", "T", 13.0, 2025, 40, 5120, 0.0078, 100, 5000],
        ["meta-llama/Llama-3.2-1B", "T", 1.0, 2024, 16, 2048, 0.0078, 128, 9000],
        ["meta-llama/Llama-3.2-3B", "T", 3.0, 2024, 28, 3072, 0.0091, 128, 9000],
        ["meta-llama/Llama-3.1-8B", "T", 8.0, 2024, 32, 4096, 0.0078, 128, 15000],
        ["mistralai/Ministral-3-3B-Base-2512", "T", 3.0, 2025, 26, 3072, 0.0084, 131, 3000],
        ["mistralai/Ministral-3-8B-Base-2512", "T", 8.0, 2025, 34, 4096, 0.0083, 131, 3000],
        ["mistralai/Ministral-3-14B-Base-2512", "T", 14.0, 2025, 40, 5120, 0.0078, 131, 3000],
        ["Qwen/Qwen3-0.6B-Base", "T", 0.6, 2025, 28, 1024, 0.027, 151, 36000],
        ["Qwen/Qwen3-1.7B-Base", "T", 1.7, 2025, 28, 2048, 0.014, 151, 36000],
        ["Qwen/Qwen3-4B-Base", "T", 4.0, 2025, 36, 2560, 0.014, 151, 36000],
        ["Qwen/Qwen3-8B-Base", "T", 8.0, 2025, 36, 4096, 0.0088, 151, 36000],
        ["Qwen/Qwen3-14B-Base", "T", 14.0, 2025, 40, 5120, 0.0078, 151, 36000],
        ["state-spaces/mamba-130m-hf", "mamba", 0.13, 2024, 24, 768, 0.031, 50, 300],
        ["state-spaces/mamba-370m-hf", "mamba", 0.37, 2024, 48, 1024, 0.047, 50, 300],
        ["state-spaces/mamba-790m-hf", "mamba", 0.79, 2024, 48, 1536, 0.031, 50, 300],
        ["state-spaces/mamba-1.4b-hf", "mamba", 1.4, 2024, 48, 2048, 0.023, 50, 300],
        ["state-spaces/mamba-2.8b-hf", "mamba", 2.8, 2024, 64, 2560, 0.025, 50, 300],
        ["AntonV/mamba2-130m-hf", "mamba", 0.13, 2024, 24, 768, 0.031, 50288, 300],
        ["AntonV/mamba2-370m-hf", "mamba", 0.37, 2024, 48, 1024, 0.047, 50288, 300],
        ["AntonV/mamba2-780m-hf", "mamba", 0.78, 2024, 48, 1536, 0.031, 50288, 300],
        ["AntonV/mamba2-1.3b-hf", "mamba", 1.3, 2024, 48, 2048, 0.023, 50288, 300],
        ["AntonV/mamba2-2.7b-hf", "mamba", 2.7, 2024, 64, 2560, 0.025, 50288, 300],
        ["fla-hub/rwkv7-191M-world", "rwkv", 0.191, 2025, 12, 768, 0.016, 65, 1600],
        ["fla-hub/rwkv7-0.4B-world", "rwkv", 0.4, 2025, 24, 1024, 0.023, 65, 3100],
        ["fla-hub/rwkv7-1.5B-world", "rwkv", 1.5, 2025, 24, 2048, 0.012, 65, 5600],
        ["fla-hub/rwkv7-2.9B-world", "rwkv", 2.9, 2025, 32, 2560, 0.013, 65, 5600],
        ["fla-hub/rwkv7-7.2B-g0a", "rwkv", 7.2, 2025, 32, 4096, 0.0078, 65, 5600],
    ],
    columns=[
        "model_name",
        "architecture",
        "n_params",
        "release_year",
        "depth",
        "width",
        "ratio",
        "vocab_size",
        "training_tokens",
    ],
)

to_keep = metadata.model_name.tolist()

feature_rename = {
    "subj_NUM": "Subject number",
    "embed_NUM": "Embedded number",
    "obj_NUM": "Object number",
    "sentence_RC_attached": "Attachment site",
    "verb_ZIPF": "Verb lemma",
    "obj_ZIPF": "Object Zipf",
    "embed_ZIPF": "Embed. Zipf",
    "sentence_CLAUSE": "Relative clause type",
    "obj_GEN": "Object gender",
    "embed_GEN": "Embedded gender",
    "subj_GEN": "Subject gender",
    "subj_ZIPF": "Subject Zipf",
}

feature_order = [
    "Relative clause type",
    "Attachment site",
    "Verb lemma",
    "Subject number",
    "Subject gender",
    "Embedded number",
]

families_rename = {
    "gpt": "gpt-neo",
    "Llama": "Llama-3.2",
    "Ministral": "Ministral-3",
    "OLMo": "OLMo-2",
    "rwkv7": "RWKV7",
}

levels_rename = {
    "Relative clause type": {
        "objwho": "Object relative",
        "subjwho": "Subject relative",
    },
    "Verb lemma": {
        4.59: "see",
        4.73: "like",
    },
}


def clean_df(
    df: pd.DataFrame,
    meta_cols: tp.List[str] = ["model_name", "layer", "revision"],
) -> pd.DataFrame:
    # collapse dotted column names by keeping last element
    # e.g. "train.representations.model_name" -> "model_name"
    cols_rename = {c: c.split(".")[-1] for c in df.columns}
    if "Feature" in cols_rename:
        cols_rename["mean"] = "FI"
        df["Feature"] = df["Feature"].replace(feature_rename)
    else:
        cols_rename["mean"] = "Spearman"
    df = df.rename(columns=cols_rename | feature_rename)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if "layer" in df.columns:
        df = df[df["layer"] > 0]

    meta_cols = [c for c in meta_cols if c in df.columns]
    if len(meta_cols) == 0:
        return df

    meta = df[meta_cols].drop_duplicates()
    if "model_name" in meta_cols:
        models_meta = (
            df[["model_name"]].drop_duplicates().merge(metadata, on="model_name", how="left", validate="one_to_one")
        )
        models = models_meta["model_name"].str.split("/").str[-1]
        for to_remove in [
            "-deduped",
            "-world",
            "-hf",
            "-g0a",
            "-Base",
            "-0724",
            "-1124",
            "-0425",
            "-2512",
            "-HF",
            "-pt",
        ]:
            models = models.str.replace(to_remove, "")
        # Convert to categories to keep original order in plots
        models_meta["model"] = pd.Categorical(models, categories=models.unique())
        families = models.str.split("-").str[0]
        families = families.replace(families_rename)
        models_meta["family"] = pd.Categorical(families, categories=families.unique())
        rank = models_meta.model.cat.codes
        rank -= rank.groupby(families).transform("min")
        models_meta["rank"] = rank
        rel_rank = rank / rank.groupby(families).transform("max")
        models_meta["rel. rank"] = rel_rank.round(1).fillna(1)

        meta = meta.merge(models_meta)
        if "layer" in meta_cols:
            # Add normalized layers
            max_layer = meta.groupby("model")["layer"].transform("max")
            meta["rel. layer"] = (meta["layer"] - 1) / (max_layer - 1)

    return df.merge(meta), meta


def load_df(
    path: str | Path,
    meta_cols: tp.List[str] = ["model_name", "layer", "revision"],
    to_keep: tp.List[str] | None = None,
) -> pd.DataFrame | tp.Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    if to_keep is not None:
        model_name_col = [c for c in df.columns if c.endswith("model_name")][0]
        df = df[df[model_name_col].isin(to_keep)]
    return clean_df(df, meta_cols=meta_cols)


def load_rsa(
    path: str | Path,
    meta_cols: tp.List[str] = ["model_name", "layer", "revision"],
) -> tp.List[pd.DataFrame]:
    rsa = pd.read_parquet(path)
    df_1 = rsa[["spearman", "model_1", "layer_1"]]
    df_2 = rsa[["model_2", "layer_2"]]
    df_1.columns = ["spearman", "model_name", "layer"]
    df_2.columns = ["model_name", "layer"]
    df_1["idx"] = df_1.index
    df_2["idx"] = df_2.index
    df_1, meta = clean_df(df_1, meta_cols=meta_cols)
    df_2, _ = clean_df(df_2, meta_cols=meta_cols)
    df = df_1.merge(df_2, on=["idx"], suffixes=("", "_2")).drop(columns=["idx"])
    df["cv"] = df.groupby([c for c in df.columns if c != "spearman"]).cumcount()
    df = df.pivot(
        index=["cv"] + meta.columns.tolist(),
        columns=[c + "_2" for c in meta.columns],
        values="spearman",
    )
    df.columns.names = [n.replace("_2", "") for n in df.columns.names]

    dfs = []
    for _, d in df.groupby(level="cv"):
        d = d.droplevel(level=0)
        d = d.loc[d.columns]
        d = (d + d.T) / 2
        d = d.clip(-1, 1)
        dfs.append(d)

    return dfs


def select_top_features(
    df: pd.DataFrame,
    n_largest: int = 4,
    group_by_max: tp.List[str] = ["model_name", "revision"],
    group_by_mean: tp.List[str] | None = ["model_name", "layer", "revision"],
    target: str = "FI",
    threshold: float = 0.15,
) -> tp.List[str]:
    if group_by_mean is not None:
        group_by_mean = [g for g in group_by_mean if g in df.columns]
        df = df.groupby(group_by_mean + ["Feature"])[target]
        df = df.mean().reset_index()

    group_by_max = [g for g in group_by_max if g in df.columns]
    df = df.groupby(group_by_max + ["Feature"])[target].max()

    top_features = df.groupby(group_by_max).nlargest(n_largest)
    top_features = top_features.sort_values(ascending=False).reset_index(level=-1)
    top_features = top_features[top_features["FI"] > threshold]

    top_features = set(top_features["Feature"])
    return [f for f in feature_order if f in top_features]


def smooth_fi_by_layer(
    df: pd.DataFrame,
    sigma: float = 1.0,
    fi_col: str = "FI",
    gb_cols: tp.List[str] = ["model_name", "revision", "cv", "Feature"],
    layer_col: str = "layer",
    overwrite: bool = True,
    output_col: str = "FI_smooth",
) -> pd.DataFrame:
    """Smooth FI along layer for each model using a Gaussian filter."""
    df_out = df.copy()
    gb_cols = [c for c in gb_cols if c in df_out.columns]
    target_col = fi_col if overwrite else output_col
    df_out = df_out.sort_values(gb_cols + [layer_col])
    df_out[target_col] = df_out.groupby(gb_cols)[fi_col].transform(gaussian_filter, sigma=sigma)

    return df_out


def remove_unused_categories(df):
    df_out = df.copy()
    cat_cols = df_out.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df_out[col] = df_out[col].cat.remove_unused_categories()
    return df_out


def lineplot2d(data, x, y, r=None, hue=None, marker=None, ax=None, band_alpha=0.4, **kwargs):
    ax = ax or plt.gca()

    palette = kwargs.pop("palette", "tab10") if hue else None

    # If marker is a string that is not a column name, use it as a fixed marker
    fixed_marker = None
    if isinstance(marker, str) and marker not in data.columns:
        fixed_marker = marker
        marker = None

    gb_cols = [c for c in [hue, marker] if c]
    groups = data.groupby(gb_cols) if gb_cols else [(None, data)]

    markers = ["o", "s", "^", "v", "D", "p", "*", "X"]
    unique_m = data[marker].unique() if marker else []

    unique_h = data[hue].unique() if hue else []
    colors = sns.color_palette(palette, n_colors=len(unique_h)) if hue else [None]
    color_map = dict(zip(unique_h, colors)) if hue else {}

    for name, grp in groups:
        xv, yv = grp[x].values, grp[y].values
        rv = grp[r].values if r else None

        if len(gb_cols) == 2:
            h_val, m_val = name
        else:
            h_val = name[0] if isinstance(name, tuple) else name
            m_val = h_val

        plot_kwargs = kwargs.copy()
        if marker:
            plot_kwargs["marker"] = markers[list(unique_m).index(m_val) % len(markers)]
        elif fixed_marker:
            plot_kwargs["marker"] = fixed_marker

        if hue:
            plot_kwargs["color"] = color_map[h_val]

        # Use cleaner label for single-column groups
        if isinstance(name, tuple):
            label = ", ".join(map(str, name)) if len(name) > 1 else str(name[0])
        else:
            label = str(name) if name else kwargs.get("label")

        # Plot main trajectory, auto-assigning labels
        line = ax.plot(xv, yv, label=label, **plot_kwargs)[0]
        color = line.get_color()

        # Plot 2D error band
        if rv is not None:
            dx, dy = np.gradient(xv), np.gradient(yv)
            norm = np.hypot(dx, dy) + 1e-8
            nx, ny = -dy / norm, dx / norm

            x_tube = np.concatenate([xv + nx * rv, (xv - nx * rv)[::-1]])
            y_tube = np.concatenate([yv + ny * rv, (yv - ny * rv)[::-1]])
            band_color = mpl.colors.to_rgba(color, band_alpha)
            ax.fill(
                x_tube,
                y_tube,
                facecolor=band_color,
                edgecolor="none",
            )

    if gb_cols:
        ax.legend(title=", ".join(gb_cols))

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax


def pca_lineplot2d(
    df: pd.DataFrame,
    x: str = "PC1",
    y: str = "PC2",
    gb_cols: tp.List[str] = ["family", "model", "layer"],
    hue: str = "model",
    marker: str = "family",
    ax=None,
    **kwargs,
) -> mpl.axes.Axes:
    df_agg = df.groupby(gb_cols).mean(numeric_only=True)
    x_std = df.groupby(gb_cols)[x].std().fillna(0)
    y_std = df.groupby(gb_cols)[y].std().fillna(0)
    df_agg["radius"] = np.sqrt(x_std**2 + y_std**2)

    return lineplot2d(
        data=df_agg.reset_index(),
        x=x,
        y=y,
        r="radius",
        hue=hue,
        marker=marker,
        ax=ax,
        **kwargs,
    )
