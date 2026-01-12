import warnings

from .baselines import DecodingBaseline, EncodingBaseline
from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .feature_importance import FeatureImportance
from .pairwise_dataloader import PairwiseDataloaderBuilder
from .reduce_dimensions import ReduceDimensions
from .sentence_representations import SentenceRepresentations
from .spd_matrix_learner import SPDMatrixLearnerBuilder
from .trainer import Trainer
from .word_representations import WordRepresentations

warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = [
    "FeatureImportance",
    "DecodingBaseline",
    "EncodingBaseline",
    "Dataset",
    "EstimateCorrelations",
    "PairwiseDataloaderBuilder",
    "ReduceDimensions",
    "SentenceRepresentations",
    "SPDMatrixLearnerBuilder",
    "Trainer",
    "WordRepresentations",
]
