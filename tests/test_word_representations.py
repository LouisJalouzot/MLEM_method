import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.word_representations import WordRepresentations


def test_word_representations():
    words = pd.DataFrame(
        {
            "word": ["Mary", "sneezes", "Have", "John", "plagiarized", "?"],
            "sentence_id": [0, 0, 1, 1, 1, 1],
        }
    )
    sentences = words.groupby("sentence_id").word.apply(lambda x: " ".join(x))
    sentences = sentences.str.replace(" ?", "?").tolist()

    for model_name in ["bert-base-uncased", "gpt2"]:
        wr = WordRepresentations(
            model_name=model_name,
            token_aggregation="mean",
            layer=5,
            infra={"folder": None},
        )
        embeddings = wr.compute_representations(words)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        model.eval()
        with torch.no_grad():
            hidden_states = model(**inputs).hidden_states[5]

        if model_name == "bert-base-uncased":
            embeddings_true = [
                hidden_states[0, 1],  # Mary
                hidden_states[0, 2:5].mean(0),  # Sneezes
                hidden_states[1, 1],  # Have
                hidden_states[1, 2],  # John
                hidden_states[1, 3:7].mean(0),  # Plagiarized
                hidden_states[1, 7],  # ?
            ]
        elif model_name == "gpt2":
            embeddings_true = [
                hidden_states[0, 0],  # Mary
                hidden_states[0, 1:4].mean(0),  # Sneezes
                hidden_states[1, 0],  # Have
                hidden_states[1, 1],  # John
                hidden_states[1, 2:4].mean(0),  # Plagiarized
                hidden_states[1, 4],  # ?
            ]

        for e, e_true in zip(embeddings, embeddings_true):
            assert torch.allclose(e, e_true, atol=1e-5)
            # Reduced tolerance to 1e-5 because of floating point imprecision on GPU
