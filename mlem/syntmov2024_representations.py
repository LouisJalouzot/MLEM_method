"""SyntMov2024 fMRI representations for MLEM encoding analysis.

Loads preprocessed BOLD NIfTI, applies HRF delay, resamples, and masks.
"""

import typing as tp

if tp.TYPE_CHECKING:
    import torch

from pathlib import Path

import numpy as np
import pandas as pd
from exca import MapInfra
from loguru import logger
from pydantic import ConfigDict, Field

from mlem.dataset import Dataset
from mlem.syntmov2024_dataset import SyntMov2024Dataset
from mlem.utils import BaseModel


class SyntMov2024Representations(BaseModel):
    """fMRI BOLD representations for SyntMov2024 encoding analysis."""

    dataset: Dataset = Field(default_factory=lambda: SyntMov2024Dataset())
    level: tp.Literal["syntmov2024"] = "syntmov2024"

    target_resolution: float = 4.0
    template_threshold: float = 0.2

    map_infra: MapInfra = MapInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @map_infra.apply(item_uid=lambda x: Path(x[0]).name, cache_type="NumpyArray")
    def preprocess(
        self, runs: tp.Iterable[tp.Tuple[str, pd.DataFrame]]
    ) -> tp.Iterator[np.ndarray]:
        """Load BOLD for each run and extract representations for all stimuli.

        Args:
            runs: Iterable of (run_id, run_df) from df.groupby("run").

        Yields:
            Array of shape (n_stimuli_in_run, n_voxels) for each run.
        """
        from ants import image_read, resample_image_to_target
        from ants.utils import from_nibabel_nifti
        from nilearn.datasets import load_mni152_brain_mask

        mask = load_mni152_brain_mask(resolution=4.2)
        mask.header.set_xyzt_units(xyz="mm")
        mask = from_nibabel_nifti(mask)
        mask_numpy = mask.numpy() > 0
        for bold_file, run_df in runs:
            # (n_trs, n_voxels)
            img = image_read(bold_file)
            img = resample_image_to_target(
                img, mask, interp_type="bSpline", imagetype=3
            )
            img = img.numpy()[mask_numpy].T

            starts = run_df["start_frame"].values
            n_volumes = run_df["n_volumes"].iloc[0]
            # (n_stimuli, n_volumes)
            indices = np.tile(np.arange(n_volumes), (starts.shape[0], 1))
            indices += starts[:, None]
            # (n_stimuli, n_volumes, n_voxels)
            img = img[indices]
            # (n_stimuli, n_voxels)
            img = img.mean(axis=1)

            yield img

    def forward(self) -> np.ndarray:
        """Load and process fMRI data for all stimuli."""
        # Group by run and concatenate results
        result = np.concatenate(
            list(self.preprocess(self.dataset.df.groupby("bold_file")))
        )
        logger.info(f"Extracted representations: {result.shape}")
        return result

    def __call__(self) -> "torch.Tensor":
        import torch

        return torch.from_numpy(self.forward())
