"""SyntMov2024 Dataset for MLEM encoding analysis.

Loads BIDS-formatted events from the SyntMov2024 syntactic movement study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mlem.dataset import Dataset

# Feature columns extracted from trial_type
FEATURE_COLS = [
    "n_chars",
    "n_words",
    "n_args",
    "has_np",
    "has_wh",
    "has_clitic",
    "has_verb",
    "is_unaccusative",
    "is_unergative",
    "is_transitive",
    "is_ditransitive",
    "is_question",
    "is_declarative",
    "has_locative",
    "total_movements",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract linguistic features from trial_type/stim columns."""
    # Filter out controls and motor targets
    mask = ~df["trial_type"].str.contains("Controls|target", case=False, na=False)
    df = df[mask].copy()
    if df.empty:
        return df

    # Text features
    clean_stim = (
        df["stim"]
        .str.replace(r"-t-", " ", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    df["n_chars"] = df["stim"].str.len()
    df["n_words"] = clean_stim.str.split().str.len()

    # Arguments from trial_type (e.g., "2Ag", "3Ag")
    df["n_args"] = df["trial_type"].str.extract(r"(\d)Ag")[0].fillna(1).astype(int)

    # Movement types (boolean as string)
    tt = df["trial_type"]
    df["has_np"] = tt.str.contains(r"NP|Unacc", case=False, na=False)
    df["has_wh"] = tt.str.contains(r"Wh", case=False, na=False)
    df["has_clitic"] = tt.str.contains(r"Cl", case=False, na=False)
    df["has_verb"] = tt.str.contains(r"V|Qinv", case=False, na=False)

    # Sentence types
    df["is_unaccusative"] = tt.str.contains(r"Unacc", case=False, na=False)
    df["is_unergative"] = tt.str.contains(r"Unerg", case=False, na=False)
    df["is_transitive"] = tt.str.contains(r"Trans", case=False, na=False)
    df["is_ditransitive"] = df["n_args"] >= 3
    df["is_question"] = tt.str.contains(r"ynQ|WhQ|Qinv", case=False, na=False)
    df["is_declarative"] = tt.str.contains(r"Decl", case=False, na=False)
    df["has_locative"] = tt.str.contains(r"loc", case=False, na=False)

    # Complexity: total movements
    movement_cols = ["has_np", "has_wh", "has_clitic", "has_verb"]
    df["total_movements"] = df[movement_cols].sum(axis=1)

    return df


class SyntMov2024Dataset(Dataset):
    """BIDS dataset loader for SyntMov2024 syntactic movement fMRI study."""

    bids_root: str = "data/syntmov2024"
    sub: str = "17"
    runs: list[str] | None = None
    space: str = "MNI152NLin2009cAsym"

    # fMRI timing parameters
    hrf_delay: float = 4.0
    n_volumes: int = 3
    tr: float | None = None

    @property
    def level(self) -> str:
        return "sentence"

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            self._features = np.array(FEATURE_COLS, dtype=str)
        return self._features

    def read(self, only_columns: bool = False) -> pd.DataFrame | np.ndarray:
        """Read events and extract features, validating against fMRI availability."""
        import nibabel as nib

        if only_columns:
            return np.array(FEATURE_COLS)

        subj_root = Path(self.bids_root) / f"sub-{self.sub}"
        if self.runs is not None:
            pattern = f"**/*task-syntmov_*run-{self.runs[0]}_*"
        else:
            pattern = "**/*task-syntmov_*"
        event_pattern = pattern + "events.tsv"
        bold_pattern = pattern + f"space-{self.space}_*bold.nii.gz"
        event_files = sorted(subj_root.glob(event_pattern))
        bold_files = sorted(subj_root.glob(bold_pattern))
        assert len(event_files) == len(bold_files), (
            "Different number of events and bold files in "
            f"{subj_root} with patterns {event_pattern} and {bold_pattern}"
        )
        logger.info(f"Found {len(event_files)} runs")
        for event_file, bold_file in zip(event_files, bold_files):
            logger.trace(f"Matching {event_file} with {bold_file}")

        dfs, total_dropped = [], 0
        for event_file, bold_file in zip(event_files, bold_files):
            df = pd.read_csv(event_file, sep="\t")

            # Remove .gz extension and replace .nii with .json
            json_file = bold_file.with_suffix("").with_suffix(".json")
            with open(json_file) as jf:
                meta = json.load(jf)
                tr = self.tr if self.tr is not None else meta.get("RepetitionTime")
                if tr is None:
                    raise ValueError(
                        f"TR not specified and RepetitionTime not found for {bold_file}"
                    )
                start_time, delay_time = (
                    meta.get("StartTime", 0.0),
                    meta.get("DelayTime", 0.0),
                )
                logger.trace(
                    f"TR={tr}, start_time={start_time}, delay_time={delay_time} for {bold_file}"
                )

            n_vols = nib.load(bold_file).shape[-1]
            adjusted_onsets = df["onset"] + start_time + delay_time + self.hrf_delay
            df["bold_file"] = bold_file
            df["start_frame"] = (adjusted_onsets / tr).astype(int)
            df["end_frame"] = df["start_frame"] + self.n_volumes

            valid = (df["start_frame"] >= 0) & (df["end_frame"] < n_vols)
            n_dropped = (~valid).sum()
            if n_dropped > 0:
                logger.trace(
                    f"{n_dropped} / {len(df)} stimuli without valid corresponding volumes in {bold_file}"
                )
                total_dropped += n_dropped
            dfs.append(df[valid])

        df = extract_features(pd.concat(dfs, ignore_index=True))

        logger.info(
            f"{total_dropped} / {len(df) + total_dropped} stimuli without valid corresponding volumes"
        )

        return df
