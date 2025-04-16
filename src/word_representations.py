import string
import typing as tp

import neuralset as ns
import numpy as np
import pandas as pd
import torch
from exca import MapInfra
from neuralset.features import HuggingFaceText
from pydantic import ConfigDict

from src.utils import BaseModel


def make_sentence(words):
    s = ""
    cum_s = []
    indices = []
    for word in words:
        if word in string.punctuation:
            s = s.strip()
        indices.append(len(s))
        s += word
        cum_s.append(s)
        s += " "

    return (s.strip(), indices, cum_s)


def df_from_words(words: pd.DataFrame) -> pd.DataFrame:
    df = words.copy()
    df["sentence"] = df.groupby("sentence_id").word.transform(
        lambda words: make_sentence(words)[0]
    )
    df["sentence_char"] = df.groupby("sentence_id").word.transform(
        lambda words: make_sentence(words)[1]
    )
    df["context"] = df.groupby("sentence_id").word.transform(
        lambda words: make_sentence(words)[2]
    )
    df = df.rename(columns={"word": "text"})
    df["timeline"] = "svo_word_level"
    df["language"] = "en"
    df["type"] = "Word"
    df["start"] = df.index
    df["duration"] = 0.5
    df = ns.segments.validate_events(df)

    return df


class WordRepresentations(BaseModel):
    level: tp.Literal["word"] = "word"
    model_name: str = "gpt2"
    token_aggregation: str = "mean"
    batch_size: int = 32
    layer: float = 5 / 13
    units: tp.List[int] | None = None
    norm: tp.Optional[int] | None = None

    _device: tp.Optional[str] = "auto"
    infra: MapInfra = MapInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def compute_representations(self, words: pd.DataFrame) -> torch.Tensor:
        # Does not perform bidirectionnal context!
        feature = HuggingFaceText(
            contextualized=True,
            token_aggregation="mean",
            model_name=self.model_name,
            device=self._device,
            batch_size=self.batch_size,
            layers=self.layer,
            infra=self.infra,
            cache_all_layers=True,
        )
        df = df_from_words(words)
        events = feature._events_from_dataframe(df)
        data = feature._get_timed_arrays(
            events, start=0, duration=df.stop.max()
        )
        data = np.array([ta.data for ta in data])

        return torch.Tensor(data)
