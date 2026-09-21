import numpy as np
import pytest
import torch

from mlem_method.feature_importance import compute_feature_importance
from mlem_method.pairwise_dataloader import PairwiseDataloader
from mlem_method.simulation import RandomMlpSimulation
from mlem_method.utils import encode_df


def generate(n=24, noise=0.25):
    simulation = RandomMlpSimulation(n=n, n_numeric=4, category_cardinalities=(4, 4, 4), d=32, noise=noise)
    Z, groups = encode_df(simulation.make_df(3), simplex=True)
    Y = simulation.make_Y(Z, groups, 3, signed=True)
    return simulation, Z, Y


def test_random_mlp_clean_and_noisy_outputs():
    noisy, Z, Y = generate()
    clean, _, Y0 = generate(noise=0)

    assert Y.shape == (24, 32)
    assert torch.equal(noisy.transform(Z), noisy.Y0)
    assert torch.equal(noisy.Y0, clean.Y0)
    assert not torch.equal(Y, Y0)


def test_random_mlp_is_pointwise_across_sample_sizes():
    small, Z_small, _ = generate(n=12)
    large, Z_large, _ = generate(n=24)

    # The MLP map is pointwise and shared across sample sizes.
    assert torch.equal(small.transform(Z_small), large.transform(Z_small))
    assert torch.equal(small.transform(Z_small), small.Y0)
    assert torch.allclose(large.transform(Z_large[:12]), large.Y0[:12])
    # MinMax scaling depends on the sample; pointwise inputs differ across sample sizes.
    assert not torch.equal(Z_small, Z_large[:12])


class RecordingScore:
    maximize = True

    def __init__(self, original, interaction=0):
        self.original = original
        self.interaction = interaction
        self.inputs = []

    def __call__(self, Z, i, j):
        self.inputs.extend(Z.clone().unbind(0))
        changed = (Z != self.original).any(dim=1)
        first, second = changed[:, :2].any(dim=-1).double(), changed[:, 2:].any(dim=-1).double()
        score = 1 - 0.2 * first - 0.3 * second - self.interaction * first * second
        return score[:, None].expand(-1, len(i))


@pytest.fixture
def recording_score(monkeypatch):
    monkeypatch.setattr(
        "mlem_method.feature_importance.predict_pairs", lambda model, loader, X, left, right: model(X, left, right)
    )
    monkeypatch.setattr("mlem_method.feature_importance.get_metric", lambda name: (lambda pred, y: pred.mean(-1), True))


def make_loader():
    values = torch.arange(8, dtype=torch.float32)
    X = torch.stack([values, -values, values.square()], dim=1)
    Y = values[:, None]
    return X, PairwiseDataloader(X, Y, n_pairs=32, min_max_scale=False, signed=True, seed=0)


@pytest.mark.parametrize("perturbations_per_eval", [1, 2, 32])
def test_stimulus_permutation_keeps_categorical_block_together(recording_score, perturbations_per_eval):
    X, loader = make_loader()
    model = RecordingScore(X)
    compute_feature_importance(
        model, loader, np.array(["c", "c", "x"]), n_perm=1, perturbations_per_eval=perturbations_per_eval
    )

    permuted_c = model.inputs[2]
    original_rows = {tuple(row.tolist()) for row in X[:, :2]}
    assert all(tuple(row.tolist()) in original_rows for row in permuted_c[:, :2])
    assert torch.equal(permuted_c[:, 2], X[:, 2])
    assert torch.equal(model.inputs[-1][:, :2], permuted_c[:, :2])
    assert torch.equal(model.inputs[-1][:, 2], model.inputs[3][:, 2])


@pytest.mark.filterwarnings("ignore:Precision loss occurred")
@pytest.mark.parametrize("perturbations_per_eval", [1, 2, 32])
def test_permutation_interaction_algebra(recording_score, perturbations_per_eval):
    X, loader = make_loader()
    additive, _ = compute_feature_importance(
        RecordingScore(X), loader, np.array(["c", "c", "x"]), n_perm=2, perturbations_per_eval=perturbations_per_eval
    )
    X, loader = make_loader()
    interacting, _ = compute_feature_importance(
        RecordingScore(X, interaction=0.4), loader, np.array(["c", "c", "x"]), n_perm=2,
        perturbations_per_eval=perturbations_per_eval,
    )

    assert abs(additive.loc[additive.Order == "interaction", "mean"].item()) < 1e-7
    assert abs(interacting.loc[interacting.Order == "interaction", "mean"].item() - 0.4) < 1e-7
