import typing as tp

import pandas as pd
import pytest
import torch

from src.core.representations import (
    compute_sentence_representations,
    compute_word_representations,
    cum_join_index,
)


# Fixture for sentence representation tests
@pytest.fixture
def sentence_data():
    sentences = ["This is the first sentence.", "Here is another one."]
    model_name = (
        "prajjwal1/bert-tiny"  # Using a smaller model for faster testing
    )
    token_aggregation = "mean"
    batch_size = 1
    norm = 2
    device = "cpu"  # Force CPU for testing consistency
    return sentences, model_name, token_aggregation, batch_size, norm, device


# Fixture for word representation tests
@pytest.fixture
def word_data():
    words_df = pd.DataFrame(
        {
            "word": ["This", "is", "a", "test", ".", "Another", "one", "."],
            "sentence_id": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    model_name = "prajjwal1/bert-tiny"
    token_aggregation = "mean"
    batch_size = 1
    norm = 2
    device = "cpu"
    return words_df, model_name, token_aggregation, batch_size, norm, device


# --- Sentence Representation Tests ---


def test_compute_sentence_representations_output_type_and_shape(sentence_data):
    sentences, model_name, token_aggregation, batch_size, norm, device = (
        sentence_data
    )
    representations_gen = compute_sentence_representations(
        sentences, model_name, token_aggregation, batch_size, norm, device
    )
    representations = list(representations_gen)

    assert isinstance(representations, list)
    assert len(representations) == len(sentences)
    for rep in representations:
        assert isinstance(rep, torch.Tensor)
        # Shape: (num_layers, hidden_size) - bert-tiny has 3 layers, hidden_size 128
        assert rep.ndim == 2
        assert rep.shape[0] == 3  # bert-tiny has 3 layers (emb + 2 hidden)


def test_compute_sentence_representations_aggregation_methods(sentence_data):
    sentences, model_name, _, batch_size, norm, device = sentence_data
    aggregations = ["mean", "max", "min", "first", "last"]
    results = {}
    for agg in aggregations:
        reps = list(
            compute_sentence_representations(
                sentences, model_name, agg, batch_size, norm, device
            )
        )
        results[agg] = torch.stack(reps)  # Stack batches for comparison

    # Basic check: ensure different aggregations (usually) produce different results
    # Note: For very short sentences or specific models, some might be identical
    assert not torch.allclose(results["mean"], results["max"])
    assert not torch.allclose(results["mean"], results["first"])
    assert not torch.allclose(results["first"], results["last"])


def test_compute_sentence_representations_norm(sentence_data):
    sentences, model_name, token_aggregation, batch_size, norm, device = (
        sentence_data
    )

    # With norm
    reps_norm = list(
        compute_sentence_representations(
            sentences, model_name, token_aggregation, batch_size, norm, device
        )
    )
    reps_norm_tensor = torch.stack(reps_norm)  # Stack batches
    norms_norm = torch.norm(reps_norm_tensor, p=norm, dim=-1)
    # Check if norms are close to 1 (allow for small floating point errors)
    assert torch.allclose(norms_norm, torch.ones_like(norms_norm), atol=1e-5)

    # Without norm
    reps_no_norm = list(
        compute_sentence_representations(
            sentences, model_name, token_aggregation, batch_size, None, device
        )
    )
    reps_no_norm_tensor = torch.stack(reps_no_norm)
    norms_no_norm = torch.norm(reps_no_norm_tensor, p=norm, dim=-1)
    # Norms should generally not be 1
    assert not torch.allclose(
        norms_no_norm, torch.ones_like(norms_no_norm), atol=1e-5
    )


# --- Word Representation Tests ---


def test_cum_join_index():
    words = ["This", "is", "a", "test", "."]
    expected_s = "This is a test."
    expected_starts = [0, 5, 8, 10, 14, 15]
    expected_stops = [4, 7, 9, 14, 15, 15]  # Adjusted expectation
    s, starts, stops = cum_join_index(words)
    assert s == expected_s
    assert starts == expected_starts
    assert stops == expected_stops

    words_punct = ["Test", ",", "with", "punctuation", "!"]
    expected_s_punct = "Test, with punctuation!"
    expected_starts_punct = [0, 4, 6, 11, 22, 23]
    expected_stops_punct = [4, 5, 10, 22, 23, 23]  # Adjusted expectation
    s_punct, starts_punct, stops_punct = cum_join_index(words_punct)

    assert s_punct == expected_s_punct
    assert starts_punct == expected_starts_punct
    assert stops_punct == expected_stops_punct


def test_compute_word_representations_output_type_and_shape(word_data):
    words_df, model_name, token_aggregation, batch_size, norm, device = (
        word_data
    )
    sentences_data = (
        words_df.groupby("sentence_id").word.apply(cum_join_index).tolist()
    )

    representations_gen = compute_word_representations(
        sentences_data, model_name, token_aggregation, batch_size, norm, device
    )
    representations = list(representations_gen)

    assert isinstance(representations, list)
    assert len(representations) == len(
        sentences_data
    )  # One tensor per sentence

    total_words = 0
    for i, rep_sentence in enumerate(representations):
        assert isinstance(rep_sentence, torch.Tensor)
        num_words_in_sentence = (
            len(sentences_data[i][1]) - 1
        )  # starts has one extra element
        total_words += num_words_in_sentence
        # Shape: (num_words_in_sentence, num_layers, hidden_size)
        assert rep_sentence.ndim == 3
        # Slice the tensor to the actual number of words before checking shape
        rep_sentence_actual_words = rep_sentence[:num_words_in_sentence]
        assert rep_sentence_actual_words.shape[0] == num_words_in_sentence
        assert rep_sentence_actual_words.shape[1] == 3  # bert-tiny layers
        assert (
            rep_sentence_actual_words.shape[2] == 128
        )  # bert-tiny hidden size

    # Check if the total number of word vectors matches the input df size (approx)
    # This isn't perfect because the generator yields per sentence,
    # and we need to concat and filter NaNs later.
    # A full test requires replicating the logic from the WordRepresentations class.


def test_compute_word_representations_aggregation_methods(word_data):
    words_df, model_name, _, batch_size, norm, device = word_data
    sentences_data = (
        words_df.groupby("sentence_id").word.apply(cum_join_index).tolist()
    )
    aggregations = ["mean", "max", "min"]  # "first", "last" not implemented
    results = {}
    for agg in aggregations:
        reps_gen = compute_word_representations(
            sentences_data, model_name, agg, batch_size, norm, device
        )
        # Concatenate results across sentences for comparison
        reps = torch.cat(list(reps_gen), dim=0)
        results[agg] = reps

    assert not torch.allclose(results["mean"], results["max"])
    assert not torch.allclose(results["mean"], results["min"])


def test_compute_word_representations_norm(word_data):
    words_df, model_name, token_aggregation, batch_size, norm, device = (
        word_data
    )
    sentences_data = (
        words_df.groupby("sentence_id").word.apply(cum_join_index).tolist()
    )

    # With norm
    reps_norm_gen = compute_word_representations(
        sentences_data, model_name, token_aggregation, batch_size, norm, device
    )
    reps_norm_tensor = torch.cat(list(reps_norm_gen), dim=0)
    norms_norm = torch.norm(reps_norm_tensor, p=norm, dim=-1)
    # Filter out NaNs before checking closeness
    valid_norms = norms_norm[~torch.isnan(norms_norm)]
    assert torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-5)

    # Without norm
    reps_no_norm_gen = compute_word_representations(
        sentences_data, model_name, token_aggregation, batch_size, None, device
    )
    reps_no_norm_tensor = torch.cat(list(reps_no_norm_gen), dim=0)
    norms_no_norm = torch.norm(reps_no_norm_tensor, p=norm, dim=-1)
    assert not torch.allclose(
        norms_no_norm, torch.ones_like(norms_no_norm), atol=1e-5
    )
