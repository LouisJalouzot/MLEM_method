import sys

from loguru import logger
from tqdm.auto import tqdm

from src.sampling_theory import CorrelationEstimator
from src.trainer import *

logger.remove()
logger.add(sink=sys.stderr, level="DEBUG")

stimulis = Stimulis(csv_path="datasets/short_sentence.csv")
X = stimulis.encode()
ce = CorrelationEstimator()
dataset_builder = PairwiseDatasetBuilder()
dataset = dataset_builder.build(X)
_, n_pairs = ce(dataset)
Trainer(
    stimulis=stimulis,
    dataset_builder=PairwiseDatasetBuilder(n_pairs=n_pairs, gamma=1),
    device="cpu",
    infra={"folder": None},  # ".cache_results"},
).train()
