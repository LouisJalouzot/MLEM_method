from __future__ import annotations

import typing as tp

from exca import TaskInfra
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from .dataset import Dataset, SimulatedRepresentations
from .sentence_representations import SentenceRepresentations
from .utils import BaseModelSharing
from .word_representations import WordRepresentations


class ReduceDimensions(BaseModelSharing):
    dataset: Dataset = Field(default_factory=lambda: Dataset())
    representations: (
        tp.Annotated[
            SentenceRepresentations | WordRepresentations | SimulatedRepresentations,
            Field(discriminator="level"),
        ]  # Use sentence or word representations based on the specified level
        | SentenceRepresentations  # Fallback to sentence representations if not specified
    ) = Field(default_factory=lambda: SentenceRepresentations())
    n_components: int = 2
    method: tp.Literal[
        "none",
        "pca",
        "tsne",
        "umap",
        "mds",
    ] = "mds"
    distance: tp.Literal[
        "euclidean",
        "manhattan",
        "cosine",
        "correlation",
        "hamming",
        "jaccard",
    ] = "euclidean"

    n_jobs: int = -1
    verbose: bool = False
    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["representations"]
    }
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("n_jobs", "verbose")

    def model_post_init(self, __context: tp.Any) -> None:
        # Ensure dataset level and representations level match
        if hasattr(self.dataset, "level") and hasattr(self.representations, "level"):
            if self.dataset.level != self.representations.level:
                raise ValueError(
                    f"Dataset level {self.dataset.level} does not match "
                    f"representations level {self.representations.level}"
                )

    @infra.apply
    def _transform_cached(self):
        """
        Applies dimensionality reduction to the representations.
        """
        data = self.representations().numpy()
        match self.method:
            case "pca":
                from sklearn.decomposition import PCA

                model = PCA(n_components=self.n_components, random_state=0)
                proj = model.fit_transform(data)
            case "tsne":
                from sklearn.manifold import TSNE

                model = TSNE(
                    n_components=self.n_components,
                    metric=self.distance,
                    random_state=0,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                proj = model.fit_transform(data)
            case "umap":
                from umap import UMAP

                model = UMAP(
                    n_components=self.n_components,
                    metric=self.distance,
                    random_state=0,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                proj = model.fit_transform(data)
            case "mds":
                from sklearn.manifold import MDS
                from sklearn.metrics import pairwise_distances

                distance_matrix = pairwise_distances(data, metric=self.distance)
                model = MDS(
                    n_components=self.n_components,
                    dissimilarity="precomputed",
                    random_state=0,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                proj = model.fit_transform(distance_matrix)
            case "none":
                proj = data[:, : self.n_components]
            case _:
                raise ValueError(
                    f"Unsupported dimensionality reduction method: {self.method}"
                )

        return proj

    def _build_transformed_df(self, proj) -> "pd.DataFrame":
        """Helper to convert projection to DataFrame with coordinate columns."""
        import pandas as pd

        proj_df = pd.DataFrame(
            proj, columns=[f"coord_{i}" for i in range(1, self.n_components + 1)]
        )
        return pd.concat([self.dataset.df, proj_df], axis=1)

    def transform(self):
        proj = self._transform_cached()
        return self._build_transformed_df(proj)

    def transform_multiple(self, model_layers: tp.List[tp.Tuple[str, int]]):
        import pandas as pd

        results = []
        for model_name, layer in tqdm(
            model_layers, desc="Reducing dimensions for multiple representations"
        ):
            task = self.infra.clone_obj(
                representations=dict(model_name=model_name, layer=layer)
            )
            proj = task._transform_cached()
            df = self._build_transformed_df(proj)
            df["model_name"] = model_name
            df["layer"] = layer
            results.append(df)

        return pd.concat(results, ignore_index=True)
