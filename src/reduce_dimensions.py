import typing as tp

from exca import TaskInfra
from pydantic import ConfigDict, Field
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances
from umap import UMAP

from src.dataset import Dataset
from src.sentence_representations import SentenceRepresentations
from src.simulated_representations import SimulatedRepresentations
from src.utils import BaseModelSharing
from src.word_representations import WordRepresentations


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
    verbose: bool = True
    infra: TaskInfra = TaskInfra(folder=".cache")
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
                model = PCA(n_components=self.n_components, random_state=0)
                proj = model.fit_transform(data)
            case "tsne":
                model = TSNE(
                    n_components=self.n_components,
                    metric=self.distance,
                    random_state=0,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                proj = model.fit_transform(data)
            case "umap":
                model = UMAP(
                    n_components=self.n_components,
                    metric=self.distance,
                    random_state=0,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                proj = model.fit_transform(data)
            case "mds":
                distance_matrix = pairwise_distances(data, metric=self.distance)
                model = MDS(
                    n_components=self.n_components,
                    dissimilarity="precomputed",
                    random_state=0,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
                proj = model.fit_transform(distance_matrix)
            case _:
                raise ValueError(
                    f"Unsupported dimensionality reduction method: {self.method}"
                )

        return proj

    def transform(self):
        proj = self._transform_cached()
        df = self.dataset.df
        df[list(range(self.n_components))] = proj

        return df
