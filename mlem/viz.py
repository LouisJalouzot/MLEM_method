import typing as tp
from collections import defaultdict
from pathlib import Path

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

pio.templates.default = "simple_white"
n_params = defaultdict(lambda: pd.NA)
n_params.update(
    {
        "openai-community/gpt2": 0.1,
        "openai-community/gpt2-medium": 0.4,
        "openai-community/gpt2-large": 0.8,
        "openai-community/gpt2-xl": 2.0,
        "EleutherAI/pythia-14m-deduped": 0.014,
        "EleutherAI/pythia-70m-deduped": 0.07,
        "EleutherAI/pythia-160m-deduped": 0.16,
        "EleutherAI/pythia-410m-deduped": 0.41,
        "EleutherAI/pythia-1b-deduped": 1.0,
        "EleutherAI/pythia-1.4b-deduped": 1.4,
        "EleutherAI/pythia-2.8b-deduped": 2.8,
        "EleutherAI/pythia-6.9b-deduped": 6.9,
        "EleutherAI/pythia-12b-deduped": 12.0,
        "EleutherAI/gpt-neo-125m": 0.125,
        "EleutherAI/gpt-neo-1.3B": 1.3,
        "EleutherAI/gpt-neo-2.7B": 2.7,
        "facebook/opt-125m": 0.125,
        "facebook/opt-1.3b": 1.3,
        "facebook/opt-2.7b": 2.7,
        "facebook/opt-6.7b": 6.7,
        "facebook/opt-13b": 13.0,
        "google/gemma-3-270m": 0.27,
        "google/gemma-3-1b-pt": 1.0,
        "google/gemma-3-4b-pt": 4.0,
        "google/gemma-3-12b-pt": 12.0,
        "allenai/OLMo-1B-0724-hf": 1.0,
        "allenai/OLMo-7B-0724-hf": 7.0,
        "allenai/OLMo-2-0425-1B": 1.0,
        "allenai/OLMo-2-1124-7B": 7.0,
        "allenai/OLMo-2-1124-13B": 13.0,
        "Qwen/Qwen2.5-0.5B": 0.5,
        "Qwen/Qwen2.5-1.5B": 1.5,
        "Qwen/Qwen2.5-3B": 3.0,
        "Qwen/Qwen2.5-7B": 7.0,
        "Qwen/Qwen2.5-14B": 14.0,
        "meta-llama/Llama-3.2-1B": 1.0,
        "meta-llama/Llama-3.2-3B": 3.0,
        "meta-llama/Llama-3.1-8B": 8.0,
        "mistralai/Ministral-3-3B-Base-2512": 3.0,
        "mistralai/Ministral-3-8B-Base-2512": 8.0,
        "mistralai/Ministral-3-14B-Base-2512": 14.0,
        "Qwen/Qwen3-0.6B-Base": 0.6,
        "Qwen/Qwen3-1.7B-Base": 1.7,
        "Qwen/Qwen3-4B-Base": 4.0,
        "Qwen/Qwen3-8B-Base": 8.0,
        "Qwen/Qwen3-14B-Base": 14.0,
        "Qwen/Qwen3.5-0.8B-Base": 0.8,
        "Qwen/Qwen3.5-2B-Base": 2.0,
        "Qwen/Qwen3.5-4B-Base": 4.0,
        "Qwen/Qwen3.5-9B-Base": 9.0,
        "state-spaces/mamba-130m-hf": 0.13,
        "state-spaces/mamba-370m-hf": 0.37,
        "state-spaces/mamba-790m-hf": 0.79,
        "state-spaces/mamba-1.4b-hf": 1.4,
        "state-spaces/mamba-2.8b-hf": 2.8,
        "AntonV/mamba2-130m-hf": 0.13,
        "AntonV/mamba2-370m-hf": 0.37,
        "AntonV/mamba2-780m-hf": 0.78,
        "AntonV/mamba2-1.3b-hf": 1.3,
        "AntonV/mamba2-2.7b-hf": 2.7,
        "RWKV/v6-Finch-1B6-HF": 1.6,
        "RWKV/v6-Finch-3B-HF": 3.0,
        "RWKV/v6-Finch-7B-HF": 7.0,
        "RWKV/v6-Finch-14B-HF": 14.0,
        "RWKV/RWKV7-Goose-World2.8-0.1B-HF": 0.1,
        "RWKV/RWKV7-Goose-World2.9-0.4B-HF": 0.4,
        "RWKV/RWKV7-Goose-World3-1.5B-HF": 1.5,
        "RWKV/RWKV7-Goose-World3-2.9B-HF": 2.9,
        "fla-hub/rwkv7-191M-world": 0.191,
        "fla-hub/rwkv7-0.4B-world": 0.4,
        "fla-hub/rwkv7-1.5B-world": 1.5,
        "fla-hub/rwkv7-2.9B-world": 2.9,
        "fla-hub/rwkv7-7.2B-g0a": 7.2,
        "LiquidAI/LFM2-350M": 0.35,
        "LiquidAI/LFM2-700M": 0.7,
        "LiquidAI/LFM2-1.2B": 1.2,
        "LiquidAI/LFM2-2.6B": 2.6,
        "google/recurrentgemma-2b": 2.0,
        "google/recurrentgemma-9b": 9.0,
    }
)


def load_df(
    path: str | Path, meta_cols: tp.List[str] = ["model_name", "layer", "revision"]
) -> pd.DataFrame | tp.Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    # collapse dotted column names by keeping last element
    # e.g. "train.representations.model_name" -> "model_name"
    cols_rename = {c: c.split(".")[-1] for c in df.columns}
    if "Feature" in cols_rename:
        cols_rename["mean"] = "FI"
    else:
        cols_rename["mean"] = "Spearman"
    df = df.rename(columns=cols_rename)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if "layer" in df.columns:
        df = df[df["layer"] > 0]

    meta_cols = [c for c in meta_cols if c in df.columns]
    if len(meta_cols) == 0:
        return df

    meta = df[meta_cols].drop_duplicates()
    if "model_name" in meta_cols:
        models_meta = df[["model_name"]].drop_duplicates()
        models_meta["n_params"] = models_meta["model_name"].replace(n_params)
        models = models_meta["model_name"].str.split("/").str[-1]
        # Convert to categories to keep original order in plots
        models_meta["model"] = pd.Categorical(models, categories=models.unique())
        families = models.str.split("-").str[0]
        models_meta["family"] = pd.Categorical(families, categories=families.unique())
        rank = models_meta.model.cat.codes
        rank -= rank.groupby(families, observed=True).transform("min")
        models_meta["rank"] = rank
        rel_rank = rank / rank.groupby(families, observed=True).transform("max")
        models_meta["rel. rank"] = rel_rank.round(1)

        meta = meta.merge(models_meta)
        if "layer" in meta_cols:
            # Add normalized layers
            max_layer = meta.groupby("model")["layer"].transform("max")
            meta["rel. layer"] = (meta["layer"] - 1) / (max_layer - 1)

    return df.merge(meta), meta


def select_top_features(
    df: pd.DataFrame,
    n_largest: int = 4,
    group_by_max: tp.List[str] = ["model_name", "revision"],
    group_by_mean: tp.List[str] | None = ["model_name", "layer", "revision"],
    target: str = "FI",
    threshold: float = 0.15,
) -> tp.List[str]:
    if group_by_mean is None:
        group_by_mean = [g for g in group_by_mean if g in df.columns]
        df = df.groupby(group_by_mean + ["Feature"], observed=True)[target]
        df = df.mean().reset_index()

    group_by_max = [g for g in group_by_max if g in df.columns]
    df = df.groupby(group_by_max + ["Feature"], observed=True)[target].max()

    top_features = df.groupby(group_by_max, observed=True).nlargest(n_largest)
    top_features = top_features.sort_values(ascending=False).reset_index(level=-1)
    top_features = top_features[top_features["FI"] > threshold]

    return list(top_features["Feature"].unique())


def scatter3d_trajectories(
    coords: pd.DataFrame,
    group_by: str = "model",
    color_col: str = "rel. layer",
    sigma: float = 4.0,
    colorscale: str = "viridis",
    line_width: int = 5,
) -> go.Figure:
    """Gaussian-smoothed 3D trajectory lines, one per group.

    ``coords`` must have columns ``x``, ``y`` (and optionally ``z``),
    ``group_by``, and ``color_col``.
    """
    fig = go.Figure()
    has_z = "z" in coords.columns

    for name, grp in coords.groupby(group_by, observed=True, sort=False):
        smooth = lambda v: gaussian_filter(v, sigma=sigma, mode="nearest")
        x, y = smooth(grp["x"].values), smooth(grp["y"].values)
        z = smooth(grp["z"].values) if has_z else np.zeros_like(x)
        color = grp[color_col].values
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                line=dict(color=color, colorscale=colorscale, width=line_width),
                marker=dict(
                    size=0,
                    color=color,
                    colorscale=colorscale,
                    showscale=False,
                    opacity=0,
                ),
                name=str(name),
                hovertemplate=f"<b>{name}</b><br>Layer: %{{customdata}}<extra></extra>",
                customdata=grp["layer"].values if "layer" in grp else None,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )
    return fig


def heatmap_grouped(
    sim: pd.DataFrame,
    color_continuous_scale: str = "inferno",
    zmin: float = 0,
    extra_sep_after: list | None = None,
) -> go.Figure:
    """Lower-triangular heatmap with grouped block tick labels.

    ``sim`` must have a MultiIndex with (model, layer) or similar.
    ``extra_sep_after`` is a list of level-0 values after which to insert
    an extra blank separator column/row.
    """
    extra_sep_after = extra_sep_after or []
    src, h_lbls, t_lbls = [], [], []
    j = 1
    for m, g in sim.groupby(level=0, sort=False, observed=True):
        n = len(g)
        src.extend(g.index.tolist() + [None])
        h_lbls.extend([f"{m} {l:>2}" for m, l in g.index] + [" " * j])
        ticks = [""] * n
        ticks[n // 2] = m
        t_lbls.extend(ticks + [" " * j])
        if m in extra_sep_after:
            src.extend([None] * 2)
            h_lbls.extend([" " * (j + 1), " " * (j + 2)])
            t_lbls.extend([" " * (j + 1), " " * (j + 2)])
            j += 2
        j += 1
    src, h_lbls, t_lbls = src[:-1], h_lbls[:-1], t_lbls[:-1]
    df_plot = sim.reindex(index=src, columns=src)
    df_plot.index = df_plot.columns = h_lbls
    df_plot = df_plot.where(np.tril(np.ones(df_plot.shape)).astype(bool))
    fig = px.imshow(
        df_plot.round(4).astype("float32"),
        color_continuous_scale=color_continuous_scale,
        zmin=zmin,
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=np.arange(len(t_lbls)), ticktext=t_lbls, ticks=""
        ),
        yaxis=dict(
            tickmode="array", tickvals=np.arange(len(t_lbls)), ticktext=t_lbls, ticks=""
        ),
    )
    return fig


def compute_dtw(i_pivot: pd.DataFrame, level: str = "model") -> pd.DataFrame:
    """Compute pairwise DTW distance matrix over sequences indexed by ``level``.

    Returns a symmetric DataFrame with NaN on the upper triangle filled in.
    """
    models = i_pivot.index.get_level_values(level).unique().tolist()
    dtw_df = pd.DataFrame(
        np.full((len(models), len(models)), np.nan), index=models, columns=models
    )
    with tqdm(total=len(models) * (len(models) + 1) // 2, desc="DTW") as pbar:
        for k, m1 in enumerate(models):
            for m2 in models[k:]:
                dtw_df.loc[m2, m1] = dtw(
                    i_pivot.xs(m1).values, i_pivot.xs(m2).values
                ).normalizedDistance
                pbar.update(1)
    vals = dtw_df.to_numpy(copy=True)
    upper = np.triu_indices_from(vals, k=1)
    vals[upper] = vals.T[upper]
    return pd.DataFrame(vals, index=dtw_df.index, columns=dtw_df.columns)


def fit_mds(
    dist: pd.DataFrame, n_components: int = 2, metric="precomputed"
) -> pd.DataFrame:
    """Fit MDS on a precomputed distance matrix, return a DataFrame of coordinates."""
    mds = MDS(
        n_components=n_components,
        metric="precomputed",
        init="classical_mds" if n_components == 2 else "random",
        n_init=1,
        random_state=0,
    )
    coords = mds.fit_transform(dist)
    return pd.DataFrame(
        coords, index=dist.index, columns=["x", "y", "z"][:n_components]
    )
