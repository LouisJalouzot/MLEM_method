# Model metadata groups

## Main-analysis groups

| Group | Features | Motivation |
|---|---|---|
| Family | Family, Normalization, Non-linearity | Normalization and non-linearity do not vary independently within families, so their separate importance cannot be identified. |
| Architecture | Architecture, Positional Encoding, Tokenizer Type | Positional encoding and tokenization are core architectural interface choices and correlate with broad architecture class. |
| Model Size | Num. Parameters, Width | Parameter count and width are nearly redundant (`r = .96`). |
| Depth | Depth | Layer count captures model shape separately from parameter count and width. |
| Data Scale | Training Tokens, Vocabulary Size, Language Focus | These properties jointly describe the scale and linguistic scope of pretraining. |
| Tied Embeddings | Tied Embeddings | Weight tying is a distinct binary design choice with low correlations to the other groups. |

## Exclusions

- **Release Date:** excluded because it is a temporal proxy for many unobserved design and training changes rather than a model property with a specific interpretation.
- **Depth / Width:** excluded because it is deterministically derived from retained properties.
- **Attention Type:** excluded because it correlates above the `.5` threshold with features assigned to multiple groups.
- **FFN / Gating Type:** excluded because it correlates above the `.5` threshold with architecture, normalization, non-linearity, and family.

These features remain available in the preliminary full-feature analysis and can be reported as sensitivity analyses.

## Correlations between groups

No averaging, canonical correlation, or SVD is used.

1. Compute one model RDM for every metadata feature.
2. Vectorize the upper triangle of each RDM, giving 903 model-pair values.
3. For every pair of groups, compute the Pearson correlation between every member RDM in the first group and every member RDM in the second group.
4. Select the correlation with the largest absolute value while retaining its sign:

\[
r_{gh} = r_{f^*k^*}, \qquad
(f^*,k^*) = \arg\max_{f\in g,\,k\in h}|r_{fk}|.
\]

The heatmap therefore remains signed, with a symmetric color scale from `-1` to `1`. The largest absolute cross-group correlation is `.414`, between Family and Positional Encoding.

These correlations are descriptive: model-pair observations sharing a model are dependent, so ordinary dyad-level correlation tests are inappropriate.

## Model-level MLEM performance

Performance is the mean Spearman correlation between predicted and observed DTW distances across the five held-out linguistic-signature folds:

| Analysis | Predictors | Spearman, mean ± SD |
|---|---|---:|
| Preliminary | All 17 metadata features | `.546 ± .008` |
| Main grouped analysis | 13 underlying features in 6 groups | `.489 ± .007` |

The grouped model retains moderate predictive performance while avoiding separate attribution to metadata properties that are redundant or structurally confounded. Grouped feature importance is computed by applying the same model-label permutation to every member RDM in a group and measuring the decrease in Spearman correlation.
