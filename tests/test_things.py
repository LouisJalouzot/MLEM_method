import mne
import numpy as np
import pandas as pd
import pytest

from mlem_method.things_dataset import THINGSDataset
from mlem_method.things_representations import THINGSFmriRepresentations, THINGSMegRepresentations


@pytest.fixture
def stimuli():
    # Deliberately unsorted, with two repetitions of the held-out image.
    return pd.DataFrame({
        "stimulus": ["b.jpg", "c.jpg", "a.jpg", "c.jpg"],
        "concept": ["b", "c", "a", "c"],
        "trial_type": ["train", "test", "train", "test"],
        "session": [1, 1, 1, 1],
    })


@pytest.fixture
def dataset(tmp_path, stimuli):
    annotations = tmp_path / "annotations"
    betas = tmp_path / "fmri/betas_csv"
    annotations.mkdir()
    betas.mkdir(parents=True)
    stimuli.to_csv(betas / "sub-01_StimulusMetadata.csv", index=False)
    pd.DataFrame({"uniqueID": ["a", "b", "c"], "size_SD": [1, 2, 3]}).to_csv(
        annotations / "property-ratings.tsv", sep="\t", index=False
    )
    pd.DataFrame({"uniqueID": ["a", "b", "c"], "COCA word freq (online)": [3, 7, 15]}).to_csv(
        annotations / "concepts-metadata_things.tsv", sep="\t", index=False
    )
    return THINGSDataset(
        root=str(tmp_path),
        features=["confound_log_coca_frequency", "size_SD"],
        infra=dict(folder=None),
    )


@pytest.fixture
def meg(dataset, stimuli, tmp_path, monkeypatch):
    """On-disk epochs read lazily, with one non-MEG channel carrying conspicuous values."""
    data = np.random.default_rng(0).normal(size=(4, 3, 31))
    # Metadata order is b, c, a, c, so stimulus c has two repetitions.
    info = mne.create_info(["MEG01", "MEG02", "MEG03", "stim"], sfreq=100, ch_types=["mag"] * 3 + ["misc"])
    epochs = mne.EpochsArray(
        np.concatenate([data, np.full((4, 1, 31), 1000.0)], axis=1),
        info,
        tmin=-0.1,
        metadata=stimuli.rename(columns={"stimulus": "image_path"}),
    )
    epochs.save(tmp_path / "epochs-epo.fif", verbose="ERROR")
    monkeypatch.setattr(
        THINGSMegRepresentations,
        "load_epochs",
        lambda self: mne.read_epochs(tmp_path / "epochs-epo.fif", preload=False, verbose="ERROR"),
    )
    return epochs


def test_feature_selection_and_split(dataset):
    assert dataset.df_features.columns.tolist() == dataset.features
    np.testing.assert_allclose(dataset.df_features, np.column_stack([np.log([4, 8, 16]), [1, 2, 3]]))
    assert dataset.stimulus_names == ["a.jpg", "b.jpg", "c.jpg"]
    assert dataset.split == ([0, 1], [2])


@pytest.mark.parametrize("features", [[]])
def test_invalid_features(features):
    with pytest.raises(ValueError):
        THINGSDataset(features=features)


def test_fmri_alignment_and_normalization(dataset, tmp_path):
    betas = tmp_path / "fmri/betas_csv"
    # Rows are voxels; columns follow the unsorted stimulus metadata.
    pd.DataFrame([[2, 8, 4, 10], [10, 15, 20, 25], [5, 7, 9, 11]]).to_hdf(
        betas / "sub-01_ResponseData.h5", key="responses"
    )
    pd.DataFrame({"IT": [1, 0, 1], "V1": [0, 0, 0]}).to_csv(
        betas / "sub-01_VoxelMetadata.csv", index=False
    )
    fmri = THINGSFmriRepresentations(dataset=dataset, infra=dict(folder=None))
    # Train-only z-scoring, canonical a/b/c order, averaged test repetitions.
    np.testing.assert_allclose(fmri().numpy(), [[1, 1], [-1, -1], [6, 1]], rtol=2e-6)
    with pytest.raises(ValueError, match="empty"):
        fmri.infra.clone_obj(roi="V1")()


def test_meg_windows_average_repetitions_and_drop_non_meg_channels(dataset, meg):
    representation = THINGSMegRepresentations(dataset=dataset, subject="1", map_infra=dict(folder=None))
    windows = [0, 1]
    all_starts, width = representation._windows(meg.times)
    starts = all_starts[windows]
    raw = meg.get_data(picks="meg")
    assert raw.shape[1] == 3  # The stim channel is not picked.
    # Dataset order is a, b, c while the epochs are b, c, a, c.
    expected = np.stack([
        np.stack([raw[indices, :, start : start + width].mean(axis=(0, -1)) for indices in ([2], [0], [1, 3])])
        for start in starts
    ])
    for got, want in zip(representation.window_data(windows), expected):
        np.testing.assert_allclose(got, want, rtol=1e-6)


def test_meg_window_cache_and_timestamps(dataset, meg, tmp_path, monkeypatch):
    meg = THINGSMegRepresentations(dataset=dataset, subject="1", map_infra=dict(folder=str(tmp_path / "cache")))
    _, second = list(meg.window_data([0, 1]))
    # At 100 Hz, the first 20-ms window contains samples at -100 and -90 ms.
    assert meg.window_times[0] == pytest.approx(-0.095)
    later = meg.map_infra.clone_obj(t=1)
    monkeypatch.setattr(THINGSMegRepresentations, "load_epochs", lambda self: pytest.fail("cache miss"))
    np.testing.assert_array_equal(later().numpy(), second)
