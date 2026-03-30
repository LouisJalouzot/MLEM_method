from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances

from mlem.viz import load_df, select_top_features, to_keep

SCRIPT_DIR = Path(__file__).parent
OUTPUT_PATH = SCRIPT_DIR / "figures" / "nearest_features.html"
INDEX_COLS = ["family", "model", "layer", "layer_idx"]
N_INTERIOR_LAYERS = 6


def sample_layers(layers: pd.Series, keep: int) -> pd.Index:
    layers = np.sort(pd.unique(layers))
    if len(layers) <= keep:
        return pd.Index(layers)
    idx = np.round(np.linspace(0, len(layers) - 1, keep)).astype(int)
    return pd.Index(layers[idx])


def build_profiles(df: pd.DataFrame, meta_cols: list[str], n_layers: int) -> pd.DataFrame:
    profiles = df.groupby(["Feature", *meta_cols], observed=True)["FI"].mean().reset_index()
    sampled = profiles.groupby("model", observed=True)["layer"].transform(
        lambda s: s.isin(sample_layers(s, keep=n_layers + 2))
    )
    profiles = profiles[sampled].copy()
    profiles["layer_idx"] = (
        profiles.groupby("model", observed=True)["layer"].rank(method="dense").astype(int)
    )
    profiles = profiles[profiles["layer_idx"].between(2, n_layers + 1)].copy()
    profiles["layer_idx"] -= 1

    return profiles.pivot(index=INDEX_COLS, columns="Feature", values="FI").sort_index()


def find_matches(profiles: pd.DataFrame) -> pd.DataFrame:
    sim = pd.DataFrame(
        pairwise_distances(profiles, metric="euclidean"),
        index=profiles.index,
        columns=profiles.index,
    )
    family = sim.index.get_level_values("family").to_numpy()
    layer_idx = sim.index.get_level_values("layer_idx").to_numpy()
    invalid = (family[:, None] == family[None, :]) | (
        layer_idx[:, None] != layer_idx[None, :]
    )
    sim = sim.mask(invalid)

    return pd.DataFrame(
        {"nearest": sim.idxmin(axis=1), "farthest": sim.idxmax(axis=1)},
        index=profiles.index,
    )


def build_plot_df(matches: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    target_cols = [f"{col}_target" for col in INDEX_COLS]
    rows = []

    for ref_idx, match_row in matches.iterrows():
        ref = profiles.loc[ref_idx]
        for match, target_idx in match_row.items():
            cur = pd.DataFrame(
                {
                    "Feature": profiles.columns,
                    "fi_ref": ref.to_numpy(),
                    "fi": profiles.loc[target_idx].to_numpy(),
                    "match": match,
                }
            )
            for col, value in zip(INDEX_COLS, ref_idx):
                cur[col] = value
            for col, value in zip(target_cols, target_idx):
                cur[col] = value
            rows.append(cur)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df, meta = load_df(SCRIPT_DIR / "0.parquet", to_keep=to_keep)
    profiles_all = build_profiles(df, list(meta.columns), n_layers=N_INTERIOR_LAYERS)
    matches = find_matches(profiles_all)

    top_features = [
        f for f in select_top_features(df, n_largest=5) if f in profiles_all.columns
    ]
    plot_df = build_plot_df(matches, profiles_all[top_features])
    max_fi = 1.1 * plot_df[["fi_ref", "fi"]].to_numpy().max()

    fig = px.scatter(
        plot_df,
        x="fi_ref",
        y="fi",
        color="Feature",
        hover_data=[
            "model",
            "model_target",
            "layer",
            "layer_target",
            "layer_idx",
            "layer_idx_target",
        ],
        facet_row="match",
        facet_col="layer_idx",
        animation_frame="model",
        category_orders={
            "match": ["nearest", "farthest"],
            "layer_idx": list(range(1, N_INTERIOR_LAYERS + 1)),
        },
        labels={"fi_ref": "Reference FI", "fi": "Matched FI"},
        range_x=[-0.1, max_fi],
        range_y=[-0.1, max_fi],
    )
    fig.add_trace(
        go.Scatter(
            x=[-0.1, max_fi],
            y=[-0.1, max_fi],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
        ),
        row="all",
        col="all",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=")[-1]))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.write_html(OUTPUT_PATH, include_plotlyjs="cdn", auto_play=False)


if __name__ == "__main__":
    main()
