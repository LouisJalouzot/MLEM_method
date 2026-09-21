from pathlib import Path

import numpy as np
import pytest

from mlem_method import FeatureImportance
from mlem_method.baselines import FRRSA


@pytest.fixture
def fi(tmp_path, monkeypatch):
    # Isolate caches; default config serialization still reads the dataset directory.
    (tmp_path / "datasets").symlink_to(Path(__file__).resolve().parents[1] / "datasets", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    return FeatureImportance(
        dataset={
            "seed": 3,
            "mahalanobis": True,
            "simulation": {"kind": "mlp", "n": 64, "n_numeric": 2, "category_cardinalities": (4,), "d": 8},
        },
        # Keep pair-budget estimation small and on CPU.
        estimate_correlations={
            "device": "cpu", "product": False, "n_trials": 2,
            "init_sample_size": 2048, "max_sample_size": 4096, "thresh": 2,
        },
        trainer={
            "kind": "frrsa",
            "dataloader_builder": {"cv": 2},
            "representations": {"level": "simulated"},
        },
        n_perm=2,
        fi_splits=("test",),
    )


@pytest.mark.parametrize("scoring", ["pearson", "spearman"])
def test_training(fi, scoring):
    trainer = fi.infra.clone_obj(trainer={"scoring": scoring}).trainer
    folds = list(trainer.train())
    assert len(folds) == 2
    np.testing.assert_array_equal(trainer.fractions, np.linspace(0.05, 1, 20))

    for model, _, _, _ in folds:
        assert isinstance(model, FRRSA)
        assert model.best_estimator_[-1].fracs in trainer.fractions
        assert np.isfinite(model.best_score_)
        # Ridge must retain every encoded coordinate, including categorical ones.
        assert model.n_features_in_ == fi.dataset.n_coordinates > fi.dataset.n_features


def test_inner_cv_has_no_stimulus_leakage(fi):
    model, _, train, _ = next(fi.trainer.train())
    # Fresh loaders have the same seed: replay the training pairs.
    left, right, *_ = train.sample(train.n_pairs, get_idx=True, only_valid=True)
    pairs = np.column_stack((left.numpy(), right.numpy()))

    for train_rows, test_rows in model.cv:
        assert len(train_rows) and len(test_rows)
        train_stimuli = set(pairs[train_rows].ravel())
        test_stimuli = set(pairs[test_rows].ravel())
        assert train_stimuli.isdisjoint(test_stimuli)


def test_feature_importance(fi):
    importance, scores, weights = fi.compute()

    # Three original features give three main effects and three interactions per fold.
    assert len(importance) == 2 * 6
    assert {"Feature", "AllFeatures", "Order", "Group", "mean", "cv", "split"} <= set(importance.columns)
    assert set(importance.Order) == {"main", "interaction"}
    assert set(importance.split) == {"test"}
    assert len(scores) == 2
    assert np.isfinite(importance["mean"]).all()
    assert np.isfinite(scores["mean"]).all()
    assert weights.empty
