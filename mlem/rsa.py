from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    import pandas as pd

from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .pairwise_dataloader import PairwiseDataloader, PairwiseDataloaderBuilder
from .sentence_representations import SentenceRepresentations
from .simulated_representations import SimulatedRepresentations
from .syntmov2024_representations import SyntMov2024Representations
from .utils import (
    BaseModelSharing,
    compute_stats,
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
    """

    dataset: Dataset = Field(default_factory=lambda: Dataset())
    estimate_correlations: EstimateCorrelations = Field(
        default_factory=lambda: EstimateCorrelations()
    )
    representations_1: tp.Annotated[
        SentenceRepresentations
        | WordRepresentations
        | SimulatedRepresentations
        | SyntMov2024Representations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())
    representations_2: tp.Annotated[
        SentenceRepresentations
        | WordRepresentations
        | SimulatedRepresentations
        | SyntMov2024Representations,
        Field(discriminator="level"),
    ] = Field(default_factory=lambda: SentenceRepresentations())
    dataloader_builder: PairwiseDataloaderBuilder = Field(
        default_factory=lambda: PairwiseDataloaderBuilder()
    )
    n_batches: int = 10
    device: tp.Optional[str] = None

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    map_infra: MapInfra = MapInfra()
    model_config: ConfigDict = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device", "map_infra")
    _shared_fields_config: tp.ClassVar[tp.Dict[str, tp.List[str]]] = {
        "dataset": ["estimate_correlations", "representations_1", "representations_2"],
    }

    @infra.apply(exclude_from_cache_uid=["device"])
    def compute(self) -> tp.List[float]:
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

        logger.info(
            f"RSA: loading representations onto {device} "
            f"(n_pairs={n_pairs}, n_batches={self.n_batches})"
        )
        Y1 = self.representations_1().to(device)
        Y2 = self.representations_2().to(device)

        dl = PairwiseDataloader(
            Y=Y1,
            Y2=Y2,
            n_pairs=n_pairs,
            distance=self.dataloader_builder.distance,
            nan_to_num=self.dataloader_builder.nan_to_num,
            min_max_scale=self.dataloader_builder.min_max_scale,
            seed=seed_from_basemodel(self),
        )

        correlations: tp.List[float] = []
        for i in range(1, self.n_batches + 1):
            # __getitem__ calls sample(n_pairs * gamma**i); with gamma=1 this
            # is always n_pairs.  Returns (Y_dist, Y2_dist) when only Y+Y2 set.
            dist1, dist2 = dl[i]
            r = spearman(dist1, dist2)
            correlations.append(float(r))
            logger.debug(f"RSA batch {i}/{self.n_batches}: r = {r:.4f}")

        logger.info(f"RSA done. Mean r = {sum(correlations) / len(correlations):.4f}")
        return correlations

    def compute_stats(self) -> "pd.DataFrame":
        """
        Run compute() and return a summary DataFrame (mean, std, CI) via
        mlem.utils.compute_stats.
        """
        return compute_stats(self.compute())

    @map_infra.apply(
        item_uid=str,
        exclude_from_cache_uid=(
            "representations_1.layer",
            "representations_2.layer",
        ),
    )
    def run_layers(
        self, layers: tp.Iterable[tp.Tuple[int, int]]
    ) -> tp.Iterator[pd.DataFrame]:
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
            rsa_for_layers = self.infra.clone_obj(
                representations_1=dict(layer=l1),
                representations_2=dict(layer=l2),
            )
            res = rsa_for_layers.compute_stats()
            res["layer_1"] = l1
            res["layer_2"] = l2
            yield res

    def run_all_layers(
        self,
        layers_1: tp.Optional[tp.Iterable[int]] = None,
        layers_2: tp.Optional[tp.Iterable[int]] = None,
    ) -> pd.DataFrame:
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

        if layers_1 is None:
            n1 = get_n_layers(self.representations_1.model_name)
            layers_1 = range(n1 + 1)
        if layers_2 is None:
            n2 = get_n_layers(self.representations_2.model_name)
            layers_2 = range(n2 + 1)

        pairs = [(l1, l2) for l1 in layers_1 for l2 in layers_2]
        logger.info(f"RSA: running {len(pairs)} layer pairs")

        all_res = []
        for res in tqdm(self.run_layers(pairs), total=len(pairs), desc="RSA Layers"):
            all_res.append(res)

        return pd.concat(all_res, ignore_index=True)
