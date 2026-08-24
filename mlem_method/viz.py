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
from joblib import Memory, Parallel, delayed
from mlem.mlem import MLEM
from plotly.colors import sample_colorscale
from scipy.ndimage import gaussian_filter
from scipy.stats import weightedtau
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["savefig.bbox"] = "tight"
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

metadata = pd.read_csv("model_metadata.csv")

to_keep = metadata.model_name.tolist()

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "think_alike/figures"
MEMORY = Memory(ROOT / ".cache/joblib", verbose=0)
MAIN_COHORT_SIZE = 44
MAIN_GROUPS = {
    "Model size": ["Num. Parameters", "Active Parameters", "Depth", "Width"],
    "Training data": ["Training Tokens", "Training Context Length", "Vocabulary Size", "Language Focus"],
    "Input/output interface": ["Tokenizer Type", "Tied Embeddings"],
    "Sequence computation": ["Positional Encoding", "Token Mixer"],
    "Block transformation": ["Normalization", "Non-linearity"],
}
GROUP_COLORS = {group: mpl.colormaps[palettes[i]](0.7) for i, group in enumerate(MAIN_GROUPS)}

feature_rename = {
    "subj_NUM": "Subject number",
    "prep_LEMMA": "Preposition lemma",
    "sentence_PP_attached": "Attachment site",
    "embed_NUM": "Embedded number",
    "embedobj_NUM": "Embedded object number",
    "embedobj_GEN": "Embedded object gender",
    "embedobj_ZIPF": "Embedded object Zipf",
    "intervener_NUM": "Intervener number",
    "intervener_GEN": "Intervener gender",
    "obj_NUM": "Object number",
    "sentence_RC_attached": "Attachment site",
    "verb_ZIPF": "Verb lemma",
    "verb_LEMMA": "Verb lemma",
    "obj_ZIPF": "Object Zipf",
    "embed_ZIPF": "Embedded Zipf",
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


def load_cohort():
    if len(metadata) != MAIN_COHORT_SIZE:
        raise ValueError(f"Expected {MAIN_COHORT_SIZE} main-cohort models in model_metadata.csv, found {len(metadata)}")
    model_metadata = metadata.copy()
    _, meta = clean_df(model_metadata[["model_name"]], meta_cols=["model_name"])
    index = pd.MultiIndex.from_frame(meta[["family", "model"]])
    features = model_metadata[[feature for members in MAIN_GROUPS.values() for feature in members]]
    X = MLEM(distance="precomputed", device="cpu")._encode_df(features)
    features_dist = (X[None] - X[:, None]).abs().clip(0, 1).nan_to_num(0)
    return model_metadata, index, features, features_dist


@MEMORY.cache
def load_distance_folds(condition):
    """Per-fold symmetric model-distance matrices for one condition, from the canonical parquets.

    Returns {"mlem": [fold_df, ...], "rsa": [fold_df, ...]}.
    """
    long_range = condition == "long_range"
    inputs = {
        "mlem": ROOT / "experiments/think_alike" / ("long_range_2/0.parquet" if long_range else "families/0.parquet"),
        "rsa": ROOT
        / "experiments/think_alike"
        / ("rsa/model_long_range_2/0.parquet" if long_range else "rsa/model/0.parquet"),
    }
    _, index, _, _ = load_cohort()

    folds_by_method = {}
    for method, input_path in inputs.items():
        if method == "rsa":
            raw = pd.read_parquet(input_path)
            left, right = [column for column in raw if column.endswith("model_name")]
            if len({left, right}) != 2 or "spearman" not in raw:
                raise ValueError(f"Unexpected RSA input schema: {input_path}")
            raw = raw[raw[left].isin(to_keep) & raw[right].isin(to_keep)]
            models = raw[left].drop_duplicates().tolist()
            pair_counts = raw.groupby([left, right]).size()
            if len(pair_counts) != MAIN_COHORT_SIZE**2 or not pair_counts.eq(5).all():
                raise ValueError(
                    f"{input_path} does not contain five observations for all {MAIN_COHORT_SIZE**2} ordered model pairs"
                )
            raw["cv"] = raw.groupby([left, right]).cumcount()
            distance_dfs = []
            for cv in sorted(raw["cv"].unique()):
                correlations = raw[raw["cv"] == cv].pivot(index=left, columns=right, values="spearman")
                correlations = correlations.loc[models, models]
                correlations = ((correlations + correlations.T) / 2).clip(-1, 1)
                distance_dfs.append(
                    pd.DataFrame(np.sqrt(2 * (1 - correlations.to_numpy())), index=index, columns=index)
                )
        else:
            i, meta_i = load_df(input_path, to_keep=to_keep)
            if i["model_name"].nunique() != MAIN_COHORT_SIZE:
                raise ValueError(f"{input_path} does not contain the complete {MAIN_COHORT_SIZE}-model main cohort")
            i_pivot = i.pivot_table(
                index=["cv", "family", "model", "layer"],
                columns="Feature",
                values="FI",
            )
            cvs = sorted(i_pivot.index.get_level_values("cv").unique())
            if len(cvs) != 5:
                raise ValueError(f"{input_path} must contain five cross-validation folds")
            local_index = pd.MultiIndex.from_frame(meta_i[["family", "model"]].drop_duplicates())
            distance_dfs = [pd.DataFrame(np.nan, index=local_index, columns=local_index) for _ in cvs]
            pbar = tqdm(
                total=len(cvs) * len(local_index) * (len(local_index) + 1) // 2, desc=f"Computing {condition} DTW"
            )
            with pbar:
                for fold, cv in enumerate(cvs):
                    for k, (family_1, model_1) in enumerate(local_index):
                        for family_2, model_2 in local_index[k:]:
                            x = i_pivot.loc[cv, family_1, model_1].values
                            y = i_pivot.loc[cv, family_2, model_2].values
                            distance_dfs[fold].loc[(family_2, model_2), (family_1, model_1)] = dtw(
                                x, y
                            ).normalizedDistance
                            pbar.update(1)

        # Align every fold onto the canonical cohort index.
        folds_by_method[method] = [
            distance.combine_first(distance.T).fillna(0.0).reindex(index=index, columns=index).fillna(0.0)
            for distance in distance_dfs
        ]
    return folds_by_method
