import warnings

from .baselines import DecodingBaseline, EncodingBaseline
from .dataset import Dataset
from .estimate_correlations import EstimateCorrelations
from .feature_importance import FeatureImportance
from .pairwise_dataloader import PairwiseDataloaderBuilder
from .reduce_dimensions import ReduceDimensions
from .sentence_representations import SentenceRepresentations
from .spd_matrix_learner import SPDMatrixLearnerBuilder
from .syntmov2024_dataset import SyntMov2024Dataset
from .syntmov2024_representations import SyntMov2024Representations
from .trainer import Trainer
from .univariate import UnivariateAnalysis
from .utils import get_n_layers
from .word_representations import WordRepresentations

warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = [
    "get_n_layers",
    "FeatureImportance",
    "DecodingBaseline",
    "EncodingBaseline",
    "Dataset",
    "EstimateCorrelations",
    "PairwiseDataloaderBuilder",
    "ReduceDimensions",
    "SentenceRepresentations",
    "SPDMatrixLearnerBuilder",
    "SyntMov2024Dataset",
    "SyntMov2024Representations",
    "Trainer",
    "UnivariateAnalysis",
    "WordRepresentations",
]
