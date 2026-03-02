# Changelog

## 2026-03-02
- **Features**
  - **New `RSA` class** in `mlem/rsa.py` for Representational Similarity Analysis: computes Spearman correlation between RDMs of two model representations via random pair sampling. Integrates with the caching pipeline (`TaskInfra`, `BaseModelSharing`) and supports HPC execution. Includes `dataset` field sharing with sub-components via `_shared_fields_config`.
  - Added `run_layers` and `run_all_layers` methods to `RSA`, leveraging `MapInfra` for efficient batch computation of RSA across multiple layer pairs.
  - **Augmented `PairwiseDataloader`** with optional `Y2` parameter: enables computing distances for two representation matrices from identical pair indices, eliminating seed-synchronisation risk. `PairwiseDataloaderBuilder.build()` updated to support `Y2` across all cv modes.

## 2026-03-01
- **Features**
  - Added `normalize_by_word` configuration parameter to `SentenceRepresentations` and `WordRepresentations`.
  - Added `subtract_word_mean_tokens` to `mlem/hidden_states.py`, allowing the subtraction of per-(position, token-type) baseline hidden state means for sentence-level data.
  - Added subtraction for per-word-type baseline representation means for word-level datasets.
- **Testing**
  - Added extensive unit testing for representation classes in `tests/test_representations.py`.
