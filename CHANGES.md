# Changelog

## Unreleased
- **Features**
  - Added `normalize_by_word` configuration parameter to `SentenceRepresentations` and `WordRepresentations`.
  - Added `subtract_word_mean_tokens` to `mlem/hidden_states.py`, allowing the subtraction of per-(position, token-type) baseline hidden state means for sentence-level data.
  - Added subtraction for per-word-type baseline representation means for word-level datasets.
- **Testing**
  - Added extensive unit testing for representation classes in `tests/test_representations.py`.
- **Dependencies**
  - Added `pytest-testmon` to project requirements for test subsetting functionality.
