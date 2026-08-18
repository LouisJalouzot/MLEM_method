import typing as tp

if tp.TYPE_CHECKING:
    import pandas as pd

import numpy as np
from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloaderBuilder
from .sentence_representations import SentenceRepresentations
from .simulation import SimulatedRepresentations
from .syntmov2024_representations import SyntMov2024Representations
from .utils import (
    BaseModelSharing,
    get_device,
    get_n_layers,
    seed_from_basemodel,
    spearman,
)
from .word_representations import WordRepresentations


class RSA(BaseModelSharing):
    """
    Representational Similarity Analysis: computes the Spearman correlation
    between the RDMs (Representational Dissimilarity Matrices) of two sets of
    representations, estimated via random pair sampling.

    Instead of building full N×N RDMs, a single PairwiseDataloader draws
    n_pairs random pairs and computes distances for both representation matrices
    from the exact same pair indices.  This is repeated n_batches times to
    obtain a distribution of Spearman r values.

    The dataset is shared across estimate_correlations, representations_1 and
    representations_2 via BaseModelSharing._shared_fields_config.

    Since RSA is symmetric, the two complete representation configurations are
    kept in a canonical order. Reversing models (and their associated layers or
    other settings) therefore produces the same cache UID and random seed.
    """

    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(default_factory=lambda: EstimateCorrelations())
    representations_1: tp.Annotated[
        SentenceRepresentations | WordRepresentations | SimulatedRepresentations | SyntMov2024Representations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())
    representations_2: tp.Annotated[
        SentenceRepresentations | WordRepresentations | SimulatedRepresentations | SyntMov2024Representations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())
    dataloader_builder: PairwiseDataloaderBuilder = Field(default_factory=lambda: PairwiseDataloaderBuilder())
    n_batches: int = 5
    device: tp.Optional[str] = None

    infra: TaskInfra = TaskInfra()
    map_infra: MapInfra = MapInfra(folder=".cache")
    layers_infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device",
        "infra",
        "layers_infra",
        "map_infra",
    )
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["estimate_correlations", "representations_1", "representations_2"],
    }

    def model_post_init(self, context):
        super().model_post_init(context)
        # Sort whole configurations so model-specific settings, including layers,
        # move with their model. Exca's clone_obj runs this before computing the UID.
        self.representations_1, self.representations_2 = sorted(
            (self.representations_1, self.representations_2),
            key=lambda representations: representations.model_dump_json(),
        )
        assert self.dataloader_builder.cv is None

    def compute(self) -> np.ndarray:
        """
        Sample n_batches batches of random pairs and return the Spearman
        correlation between the two RDMs for each batch.

        Returns
        -------
        List[float]
            Spearman r for each batch (length = n_batches).
        """
        device = self.device or get_device()

        # Retrieve stable n_pairs from the convergence estimator (cached).
        _, n_pairs = self.estimate_correlations.estimate_correlations()

        logger.debug(
            f"RSA: {self.representations_1.model_name} "
            f"layer {self.representations_1.layer} and "
            f"{self.representations_2.model_name} "
            f"layer {self.representations_2.layer} "
            f"(n_pairs={n_pairs}, n_batches={self.n_batches}, device={device})"
        )
        Y1 = self.representations_1().to(device)
        Y2 = self.representations_2().to(device)

        dl_it = self.dataloader_builder.build(
            Y=Y1,
            Y2=Y2,
            n_pairs=n_pairs,
            seed=seed_from_basemodel(self),
        )
        dl, _ = next(iter(dl_it))
        correlations = []
        for i in range(1, self.n_batches + 1):
            dist1, dist2 = dl[i]
            r = spearman(dist1, dist2)
            correlations.append(r.cpu())

        logger.debug(f"RSA done. Mean r = {sum(correlations) / len(correlations):.4f}")
        return np.array(correlations)

    @infra.apply
    def run(self) -> "pd.DataFrame":
        """Compute RSA for the configured representations."""
        import pandas as pd

        return pd.DataFrame(self.compute(), columns=["spearman"])

    @map_infra.apply(
        item_uid=str,
        exclude_from_cache_uid=(
            "representations_1.layer",
            "representations_2.layer",
        ),
        cache_type="MemmapArrayFile",
    )
    def run_layers(self, layers: tp.Iterable[tp.Tuple[int, int]]) -> tp.Iterator["pd.DataFrame"]:
        """
        Run RSA for multiple pairs of layers using MapInfra.

        Parameters
        ----------
        layers : tp.Iterable[tp.Tuple[int, int]]
            Pairs of (layer_1, layer_2) to compute RSA for.

        Returns
        -------
        tp.Iterator[pd.DataFrame]
            Results for each layer pair.
        """
        for l1, l2 in layers:
            infra_for_pair = self.layers_infra.clone_obj(
                representations_1=dict(layer=l1),
                representations_2=dict(layer=l2),
            )
            yield infra_for_pair.compute()

    def __call__(self):
        layer_1 = self.representations_1.layer
        layer_2 = self.representations_2.layer

        return self.run_all_layers((layer_1, layer_2))

    @layers_infra.apply(exclude_from_cache_uid=("representations_1.layer", "representations_2.layer"))
    def run_all_layers(
        self,
    ) -> "pd.DataFrame":
        """
        Run RSA for all pairs of layers within two ranges.

        Parameters
        ----------
        layers_1 : tp.Optional[tp.Iterable[int]]
            Layers for representations_1. If None, uses all layers of the model.
        layers_2 : tp.Optional[tp.Iterable[int]]
            Layers for representations_2. If None, uses all layers of the model.

        Returns
        -------
        pd.DataFrame
            Aggregated results for all layer pairs.
        """
        import pandas as pd

        n1 = get_n_layers(self.representations_1.model_name)
        n2 = get_n_layers(self.representations_2.model_name)

        pairs = [(l1, l2) for l1 in range(n1 + 1) for l2 in range(n2 + 1)]
        logger.info(
            f"RSA: running {len(pairs)} layer pairs for "
            f"{self.representations_1.model_name} and "
            f"{self.representations_2.model_name}"
        )

        all_res = []
        for (l1, l2), res in tqdm(zip(pairs, self.run_layers(pairs)), total=len(pairs), desc="RSA Layers"):
            res = pd.DataFrame(res, columns=["spearman"])
            res["model_1"] = self.representations_1.model_name
            res["layer_1"] = l1
            res["model_2"] = self.representations_2.model_name
            res["layer_2"] = l2
            all_res.append(res)

        return pd.concat(all_res, ignore_index=True)
