"""THINGSplus concept-property features for the shared fMRI/MEG stimuli."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field

from .dataset import Dataset

BASE_FEATURE_COLS = [
    "size_mean",
    "property_manmade_mean",
    "property_precious_mean",
    "property_lives_mean",
    "property_heavy_mean",
    "property_natural_mean",
    "property_moves_mean",
    "property_grasp_mean",
    "property_hold_mean",
    "property_be-moved_mean",
    "property_pleasant_mean",
    "property_arousal_mean",
    "size_range-length_mean",
]

CONFOUND_FEATURE_COLS = [
    "confound_image_nameability",
    "confound_image_consistency",
    "confound_image_ratings_per_image",
    "confound_image_label_ratings",
    "confound_property_ratings",
    "confound_arousal_ratings",
    "confound_size_ratings",
    "confound_size_missing",
    "confound_size_level",
    "confound_percent_known",
    "confound_word_rank",
    "confound_concreteness",
    "confound_log_coca_frequency",
    "confound_log_subtlex_frequency",
    "confound_word_meanings",
    "confound_is_bigram",
    "confound_word_length",
]

RATING_CONFOUND_SOURCE = {
    "confound_image_nameability": "image-label_nameability_mean",
    "confound_image_consistency": "image-label_consistency_mean",
    "confound_image_ratings_per_image": "image-label_ratings-per-image_mean",
    "confound_image_label_ratings": "image-label_N-ratings",
    "confound_property_ratings": "property_N-ratings",
    "confound_arousal_ratings": "property_arousal_N-ratings",
    "confound_size_ratings": "size_N-ratings",
    "confound_size_missing": "size_N-no-size",
    "confound_size_level": "size_level1",
}

CONCEPT_CONFOUND_SOURCE = {
    "confound_percent_known": "Percent_known",
    "confound_word_rank": "Rank (combining COCA/concreteness)",
    "confound_concreteness": "Concreteness (M)",
    "confound_log_coca_frequency": "COCA word freq (online)",
    "confound_log_subtlex_frequency": "SUBTLEX freq",
    "confound_word_meanings": "Number of word meanings in list",
    "confound_is_bigram": "Bigram",
}


class THINGSDataset(Dataset):
    """Shared THINGS stimuli for fMRI and MEG.

    ``features`` defaults to the semantic properties in ``BASE_FEATURE_COLS``;
    pass any ordered list of names from the annotation tables instead::

        THINGSDataset(features=["size_mean", "property_moves_mean"])
        THINGSDataset(features=[*BASE_FEATURE_COLS, *CONFOUND_FEATURE_COLS])

    ``RATING_CONFOUND_SOURCE`` and ``CONCEPT_CONFOUND_SOURCE`` rename the annotation
    columns that need a shorter name; every other requested name is read as-is from
    the two annotation tables. The concept table is optional and only read for
    lexical features.
    """

    root: str = "data/things"
    features: list[str] = Field(default_factory=lambda: BASE_FEATURE_COLS.copy(), min_length=1)

    @property
    def level(self) -> str:
        return "things"

    def read(self, only_columns=False) -> pd.DataFrame | np.ndarray:
        if only_columns:
            return np.array(self.features, dtype=str)
        return self.meta[["stimulus", "concept", *self.features]]

    @cached_property
    def meta(self) -> pd.DataFrame:
        """One row per stimulus: the 8,640 train images then the 100 test images."""
        base = Path(self.root)
        stim = pd.read_csv(base / "fmri/betas_csv/sub-01_StimulusMetadata.csv")
        train = stim.query('trial_type == "train"').sort_values("stimulus")
        test = stim.query('trial_type == "test"').drop_duplicates("stimulus").sort_values("stimulus")
        meta = pd.concat([train, test])[["stimulus", "concept", "trial_type"]]

        for table, sources in (
            ("property-ratings.tsv", RATING_CONFOUND_SOURCE),
            ("concepts-metadata_things.tsv", CONCEPT_CONFOUND_SOURCE),
        ):
            path = base / "annotations" / table
            available = pd.read_csv(path, sep="\t", nrows=0).columns
            columns = {name: sources.get(name, name) for name in self.features}
            columns = {name: column for name, column in columns.items() if column in available}
            if not columns:
                continue
            annotations = pd.read_csv(path, sep="\t", usecols=["uniqueID", *columns.values()]).rename(
                columns={column: name for name, column in columns.items()}
            )
            meta = meta.merge(
                annotations, left_on="concept", right_on="uniqueID", how="left", validate="many_to_one"
            )
        meta["confound_word_length"] = meta["concept"].str.len().astype(float)
        for name in self.features:
            meta[name] = pd.to_numeric(meta[name], errors="coerce")
            if name.startswith("confound_log_"):
                meta[name] = np.log1p(meta[name])
        missing = ~np.isfinite(meta[self.features]).all(axis=1)
        if missing.any():
            raise ValueError(f"{missing.sum()} stimuli have missing THINGS features")
        return meta

    @property
    def n_train(self) -> int:
        return int((self.meta.trial_type == "train").sum())

    @property
    def stimulus_names(self) -> list[str]:
        return self.meta.stimulus.tolist()

    @property
    def split(self) -> tuple[list[int], list[int]]:
        n = len(self.meta)
        return list(range(self.n_train)), list(range(self.n_train, n))
