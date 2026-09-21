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
        assert model.grid.best_estimator_[-1].fracs in trainer.fractions
        assert np.isfinite(model.grid.best_score_)
        # Ridge must retain every encoded coordinate, including categorical ones.
        assert model.grid.n_features_in_ == fi.dataset.n_coordinates > fi.dataset.n_features


def test_inner_cv_has_no_stimulus_leakage(fi):
    model, _, train, _ = next(fi.trainer.train())
    # Fresh loaders have the same seed: replay the training pairs.
    left, right, *_ = train.sample(train.n_pairs, get_idx=True, only_valid=True)
    pairs = np.column_stack((left.numpy(), right.numpy()))

    for train_rows, test_rows in model.grid.cv:
        assert len(train_rows) and len(test_rows)
        train_stimuli = set(pairs[train_rows].ravel())
        test_stimuli = set(pairs[test_rows].ravel())
        assert train_stimuli.isdisjoint(test_stimuli)


def test_feature_importance(fi):
    serial = fi.infra.clone_obj(perturbations_per_eval=1)
    assert fi.infra.uid() == serial.infra.uid()
    assert fi.layers_infra.uid() == serial.layers_infra.uid()
    assert fi.map_infra.uid() == serial.map_infra.uid()
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


def test_frrsa_module_matches_sklearn(fi):
    import torch

    from mlem_method.feature_importance import predict_pairs

    model, _, train, _ = next(fi.trainer.train())
    left, right, _, observed, *rest = train.sample(train.n_pairs, get_idx=True)
    X = train.X[None]
    sklearn_pred = model.grid.predict(
        train.pair_delta(left, right, X=X).square().reshape(-1, X.shape[-1]).cpu().numpy()
    ).reshape(-1)
    torch_pred = predict_pairs(model, train, X, left, right).reshape(-1).cpu()
    assert torch_pred.dtype == X.dtype and torch_pred.device.type == X.device.type
    torch.testing.assert_close(torch_pred, torch.as_tensor(sklearn_pred, dtype=torch_pred.dtype), rtol=1e-4, atol=1e-4)
