import torch
from src.utils import BaseModel
from pydantic import ConfigDict
from exca import MapInfra
import typing as tp
import neuralset as ns
import pandas as pd
from neuralset.features import HuggingFaceText
import string
import numpy as np


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


def events_from_words(words: pd.DataFrame) -> pd.DataFrame:
    events = words.copy()
    events["sentence"] = events.groupby("sentence_id").word.transform(
        lambda words: make_sentence(words)[0]
    )
    events["sentence_char"] = events.groupby("sentence_id").word.transform(
        lambda words: make_sentence(words)[1]
    )
    events["context"] = events.groupby("sentence_id").word.transform(
        lambda words: make_sentence(words)[2]
    )
    events = events.rename(columns={"word": "text"})
    events["timeline"] = "svo_word_level"
    events["language"] = "en"
    events["type"] = "Word"
    events["start"] = events.index
    events["duration"] = 0.5
    events = ns.segments.validate_events(events)

    return events


class WordRepresentations(BaseModel):
    model_name: str = "gpt2"
    token_aggregation: str = "mean"
    batch_size: int = 32
    layer: int = 5
    units: tp.List[int] = None
    norm: tp.Optional[int] = None

    _device: tp.Optional[str] = "auto"
    infra: MapInfra = MapInfra(folder=".cache")
    model_config: ConfigDict = ConfigDict(extra="forbid")

    def compute_representations(self, words: pd.DataFrame) -> torch.Tensor:
        events = events_from_words(words)
        feature = HuggingFaceText(
            contextualized=True,
            token_aggregation="mean",
            model_name=self.model_name,
            device=self._device,
            batch_size=self.batch_size,
            layer=self.layer,
            infra=self.infra,
            cache_all_layers=True,
        )
        data = feature._get_data([row for _, row in events.iterrows()])
        data = list(data)
        data = np.array(data)
        data = torch.Tensor(data)

        return data
