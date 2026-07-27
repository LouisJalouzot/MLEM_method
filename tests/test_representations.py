"""Tests for SentenceRepresentations and WordRepresentations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from exca import TaskInfra

from mlem_method.dataset import Dataset
from mlem_method.sentence_representations import SentenceRepresentations
from mlem_method.word_representations import WordRepresentations

# ── constants ─────────────────────────────────────────────────────────────────

BERT = "hf-internal-testing/tiny-random-bert"
GPT2 = "hf-internal-testing/tiny-random-gpt2"
MODELS = [pytest.param(BERT, id="bert"), pytest.param(GPT2, id="gpt2")]

# 4 sentences with shared words at different positions so normalize_by_word has effect
SENTENCES = [
    "the cat sees the dog",
    "the dog sees the cat",
    "the cat jumps the dog",
    "the dog jumps the cat",
]


def _words_rows() -> list[dict]:
    rows = []
    for sent in SENTENCES:
        cursor = 0
        for w in sent.split():
            rows.append(
                {
                    "word": w,
                    "sentence": sent,
                    "start_idx": cursor,
                    "end_idx": cursor + len(w),
                    "feat": "A",
                }
            )
            cursor += len(w) + 1
    return rows


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="session")
def sentences_csv(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("data") / "sentences.csv"
    pd.DataFrame(
        {
            "sentence": SENTENCES,
            "feat_a": ["A", "A", "B", "B"],
            "feat_b": ["X", "Y", "X", "Y"],
        }
    ).to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="session")
def words_csv(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("data") / "words.csv"
    pd.DataFrame(_words_rows()).to_csv(p, index=False)
    return str(p)


def _infra(cache_dir: str) -> TaskInfra:
    return TaskInfra(folder=cache_dir, mode="retry")


def _sr(sentences_csv, cache_dir, **kwargs) -> SentenceRepresentations:
    """Build a SentenceRepresentations with shared cache and sensible defaults."""
    defaults = dict(
        layer=1,
        device="cpu",
        infra=_infra(cache_dir),
        inner_infra=_infra(cache_dir),
    )
    defaults.update(kwargs)
    return SentenceRepresentations(dataset=Dataset(path=sentences_csv), **defaults)


def _wr(words_csv, cache_dir, **kwargs) -> WordRepresentations:
    """Build a WordRepresentations with shared cache and sensible defaults."""
    defaults = dict(
        layer=1,
        device="cpu",
        infra=_infra(cache_dir),
        inner_infra=_infra(cache_dir),
    )
    defaults.update(kwargs)
    return WordRepresentations(dataset=Dataset(path=words_csv), **defaults)


# ── SentenceRepresentations ────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", MODELS)
def test_sr_output_shape(model_name, sentences_csv, cache_dir):
    out = _sr(sentences_csv, cache_dir, model_name=model_name)()
    assert out.ndim == 2 and out.shape[0] == len(SENTENCES)


@pytest.mark.parametrize("model_name", MODELS)
def test_sr_layer_none(model_name, sentences_csv, cache_dir):
    """layer=None concatenates all transformer layers → wider than layer=1."""
    out_1 = _sr(sentences_csv, cache_dir, model_name=model_name, layer=1)()
    out_all = _sr(sentences_csv, cache_dir, model_name=model_name, layer=None)()
    assert out_all.shape[1] > out_1.shape[1]


@pytest.mark.parametrize("model_name", MODELS)
def test_sr_layer_list(model_name, sentences_csv, cache_dir):
    """layer=[0, 1] gives exactly 2× the width of layer=1."""
    w1 = _sr(sentences_csv, cache_dir, model_name=model_name, layer=1)().shape[1]
    w01 = _sr(sentences_csv, cache_dir, model_name=model_name, layer=[0, 1])().shape[1]
    assert w01 == 2 * w1


@pytest.mark.parametrize("agg", ["mean", "max", "min", "first", "last"])
def test_sr_token_aggregation(agg, sentences_csv, cache_dir):
    out = _sr(sentences_csv, cache_dir, model_name=BERT, token_aggregation=agg)()
    assert out.shape[0] == len(SENTENCES) and out.ndim == 2


def test_sr_normalize_layer0(sentences_csv, cache_dir):
    """After layer0 normalization, selecting layer 0 gives all zeros."""
    out = _sr(
        sentences_csv,
        cache_dir,
        model_name=BERT,
        layer=0,
        normalize_embeddings="layer0",
    )()
    assert torch.allclose(out, torch.zeros_like(out))


def test_sr_normalize_diff(sentences_csv, cache_dir):
    """diff normalization produces output different from baseline."""
    base = _sr(sentences_csv, cache_dir, model_name=BERT)()
    diff = _sr(sentences_csv, cache_dir, model_name=BERT, normalize_embeddings="diff")()
    assert not torch.allclose(base, diff)


@pytest.mark.parametrize("model_name", MODELS)
def test_sr_normalize_by_word(model_name, sentences_csv, cache_dir):
    """normalize_by_word=True works across token positions and completes without error."""
    normed = _sr(sentences_csv, cache_dir, model_name=model_name, normalize_by_word=True)()
    assert normed.ndim == 2 and normed.shape[0] == len(SENTENCES)


def test_sr_pca(sentences_csv, cache_dir):
    out = _sr(
        sentences_csv,
        cache_dir,
        model_name=BERT,
        pca=2,
        svd_solver="full",
    )()
    assert out.shape == (len(SENTENCES), 2)


def test_sr_noise(sentences_csv, cache_dir):
    base = _sr(sentences_csv, cache_dir, model_name=BERT, noise_level=0.0)()
    noisy = _sr(sentences_csv, cache_dir, model_name=BERT, noise_level=1.0)()
    assert not torch.allclose(base, noisy)


def test_sr_units(sentences_csv, cache_dir):
    """units=[0, 1] selects 2 hidden units, independent of token aggregation."""
    full = _sr(sentences_csv, cache_dir, model_name=BERT)()
    units = _sr(sentences_csv, cache_dir, model_name=BERT, units=[0, 1])()
    assert units.shape == (len(SENTENCES), 2)
    assert full.shape[1] > 2


# ── WordRepresentations ────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", MODELS)
def test_wr_output_shape(model_name, words_csv, cache_dir):
    rows = _words_rows()
    out = _wr(words_csv, cache_dir, model_name=model_name)()
    assert out.ndim == 2 and out.shape[0] == len(rows)


@pytest.mark.parametrize("model_name", MODELS)
def test_wr_normalize_by_word(model_name, words_csv, cache_dir):
    """After normalize_by_word, each word type's representations sum to ~0 per output unit."""
    wr = _wr(
        words_csv,
        cache_dir,
        model_name=model_name,
        layer=None,
        normalize_by_word=True,
    )
    out = wr()  # (n_words, n_layers * hidden_size), centered before layer selection
    words = np.array([r["word"] for r in _words_rows()])
    for w in np.unique(words):
        group_sum = out[words == w].sum(dim=0)
        assert group_sum.abs().max().item() < 1e-4, f"Word '{w}' group sum not zero after normalize_by_word"


def test_wr_normalize_by_word_differs(words_csv, cache_dir):
    """normalize_by_word=True produces output different from default."""
    base = _wr(words_csv, cache_dir, model_name=BERT)()
    normed = _wr(words_csv, cache_dir, model_name=BERT, normalize_by_word=True)()
    assert not torch.allclose(base, normed)


def test_wr_normalize_layer0(words_csv, cache_dir):
    """After layer0 normalization, selecting layer 0 gives all zeros."""
    out = _wr(
        words_csv,
        cache_dir,
        model_name=BERT,
        layer=0,
        normalize_embeddings="layer0",
    )()
    assert torch.allclose(out, torch.zeros_like(out))


def test_wr_normalize_diff(words_csv, cache_dir):
    base = _wr(words_csv, cache_dir, model_name=BERT)()
    diff = _wr(words_csv, cache_dir, model_name=BERT, normalize_embeddings="diff")()
    assert not torch.allclose(base, diff)


def test_wr_normalize_by_word_and_layer0_compose(words_csv, cache_dir):
    """normalize_by_word and normalize_embeddings are orthogonal and compose."""
    out = _wr(
        words_csv,
        cache_dir,
        model_name=BERT,
        layer=0,
        normalize_by_word=True,
        normalize_embeddings="layer0",
    )()
    # layer0 zeroes layer 0; normalize_by_word also shifts, but zero stays zero
    assert torch.allclose(out, torch.zeros_like(out))


def test_wr_pca(words_csv, cache_dir):
    out = _wr(words_csv, cache_dir, model_name=BERT, pca=2, svd_solver="full")()
    assert out.shape[1] == 2


def test_wr_layer_none(words_csv, cache_dir):
    w1 = _wr(words_csv, cache_dir, model_name=BERT, layer=1)().shape[1]
    wall = _wr(words_csv, cache_dir, model_name=BERT, layer=None)().shape[1]
    assert wall > w1
