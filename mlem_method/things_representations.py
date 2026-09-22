"""THINGS fMRI betas and MEG epochs as MLEM representations.

Both modalities produce one row per dataset stimulus: single trials for the main
images, super-trials (mean of the 12 repetitions) for the repeated test images.
"""

import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
from exca import MapInfra, TaskInfra
from loguru import logger
from pydantic import ConfigDict, Field

from .things_dataset import THINGSDataset
from .utils import BaseModel

if tp.TYPE_CHECKING:
    import torch


class THINGSFmriRepresentations(BaseModel):
    """Single-trial beta responses for one subject and ROI, aligned to dataset rows."""

    dataset: THINGSDataset = Field(default_factory=THINGSDataset)
    level: tp.Literal["things-fmri"] = "things-fmri"
    subject: tp.Literal["01", "02", "03"] = "01"
    roi: tp.Literal[
        "V1",
        "V2",
        "V3",
        "hV4",
        "VO1",
        "VO2",
        "LO1 (prf)",
        "LO2 (prf)",
        "TO1",
        "TO2",
        "V3b",
        "V3a",
        "lEBA",
        "rEBA",
        "lFFA",
        "rFFA",
        "lOFA",
        "rOFA",
        "lSTS",
        "rSTS",
        "lPPA",
        "rPPA",
        "lRSC",
        "rRSC",
        "lTOS",
        "rTOS",
        "lLOC",
        "rLOC",
        "IT",
    ] = "IT"
    zscore: bool = True

    infra: TaskInfra = TaskInfra(folder=".cache", mode="retry")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @property
    def is_empty(self) -> bool:
        vox = pd.read_csv(Path(self.dataset.root) / "fmri/betas_csv" / f"sub-{self.subject}_VoxelMetadata.csv")
        return self.roi not in vox.columns or not vox[self.roi].any()

    @infra.apply
    def forward(self) -> np.ndarray:
        base = Path(self.dataset.root) / "fmri/betas_csv"
        stim = pd.read_csv(base / f"sub-{self.subject}_StimulusMetadata.csv")
        vox = pd.read_csv(base / f"sub-{self.subject}_VoxelMetadata.csv")
        if self.is_empty:
            raise ValueError(f"ROI {self.roi} is empty for sub-{self.subject}")

        names = self.dataset.stimulus_names
        n_train = self.dataset.n_train
        counts = stim.stimulus.value_counts()
        assert counts.reindex(names[:n_train]).eq(1).all(), "train stimuli need exactly one trial"
        assert counts.reindex(names[n_train:]).ge(1).all(), "test stimuli need at least one trial"

        resp = pd.read_hdf(base / f"sub-{self.subject}_ResponseData.h5")  # voxels × trials
        Y = resp.drop(columns="voxel_id", errors="ignore").loc[vox[self.roi].astype(bool).to_numpy()].to_numpy()
        Y = Y.T.astype(np.float32)
        if self.zscore:
            sessions = stim.session.to_numpy()
            is_train = stim.trial_type.eq("train").to_numpy()
            for session in np.unique(sessions):
                rows = sessions == session
                Y[rows] = (Y[rows] - Y[rows & is_train].mean(axis=0)) / (Y[rows & is_train].std(axis=0) + 1e-6)

        index = {name: i for i, name in enumerate(stim.stimulus)}
        by_name = stim.stimulus.to_numpy()
        train = Y[[index[name] for name in names[:n_train]]]
        test = np.stack([Y[by_name == name].mean(axis=0) for name in names[n_train:]])
        out = np.concatenate([train, test])
        logger.info(f"THINGS fMRI sub-{self.subject} roi={self.roi}: {out.shape}")
        return out

    def __call__(self) -> "torch.Tensor":
        import torch

        return torch.from_numpy(np.ascontiguousarray(self.forward()))


class THINGSMegRepresentations(BaseModel):
    """Sensor patterns of the released THINGS MEG epochs, averaged per stimulus and time window.

    The OpenNeuro derivatives already carry the filtering, epoching and baseline
    normalization, so only the MEG channels are selected and averaged here.
    """

    dataset: THINGSDataset = Field(default_factory=THINGSDataset)
    level: tp.Literal["things-meg"] = "things-meg"
    subject: tp.Literal["1", "2", "3", "4"] = "1"
    t: int = Field(default=0, ge=0)
    bin_ms: float = Field(default=20.0, gt=0)
    stride_ms: float = Field(default=10.0, gt=0)

    map_infra: MapInfra = MapInfra(folder=".cache", version="3")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def load_epochs(self):
        import mne

        mne.set_log_level("WARNING")
        path = Path(self.dataset.root) / "meg" / f"preprocessed_P{self.subject}-epo.fif"
        return mne.read_epochs(path, preload=False)

    def _windows(self, times: np.ndarray) -> tuple[np.ndarray, int]:
        """Start indices and sample width of the overlapping time windows."""
        sfreq = 1 / np.median(np.diff(times))
        width = max(1, round(self.bin_ms / 1000 * sfreq))
        stride = max(1, round(self.stride_ms / 1000 * sfreq))
        if width > len(times):
            raise ValueError(f"window ({self.bin_ms} ms) exceeds epoch ({len(times)} samples)")
        return np.arange(0, len(times) - width + 1, stride, dtype=int), width

    @property
    def n_windows(self) -> int:
        return len(self._windows(self.load_epochs().times)[0])

    @property
    def window_times(self) -> np.ndarray:
        times = self.load_epochs().times
        starts, width = self._windows(times)
        return (times[starts] + times[starts + width - 1]) / 2

    @map_infra.apply(item_uid=str, cache_type="NumpyArray", exclude_from_cache_uid=("t",))
    def window_data(self, windows: tp.Iterable[int]) -> tp.Iterator[np.ndarray]:
        """Average the repetitions of each stimulus within the requested time windows."""
        import mne

        epochs = self.load_epochs()
        basename = epochs.metadata["image_path"].map(lambda path: Path(path).name).to_numpy()
        names = self.dataset.stimulus_names
        sel = np.flatnonzero(np.isin(basename, names))
        groups = [np.flatnonzero(basename[sel] == name) for name in names]
        assert all(len(group) for group in groups), "epochs do not cover every dataset stimulus"

        starts, width = self._windows(epochs.times)
        starts = starts[list(windows)]
        picks = mne.pick_types(epochs.info, meg=True, eeg=False, exclude="bads")
        n_channels = len(picks)

        acc = np.zeros((len(starts), len(sel), n_channels), dtype=np.float32)
        for start in range(0, len(sel), 256):
            data = epochs[sel[start : start + 256]].get_data(picks=picks)  # (epochs, channels, samples)
            for w, window_start in enumerate(starts):
                acc[w, start : start + len(data)] = data[:, :, window_start : window_start + width].mean(axis=-1)

        for mat in acc:
            yield np.stack([mat[group].mean(axis=0) for group in groups])

    def __call__(self) -> "torch.Tensor":
        import torch

        out = next(self.window_data([self.t]))
        return torch.from_numpy(np.ascontiguousarray(out))
