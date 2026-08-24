# Model metadata groups

## Main-analysis groups

| Group | Features | Motivation |
|---|---|---|
| Model size | Num. Parameters, Active Parameters, Depth, Width | Capacity and model shape. |
| Training data | Training Tokens, Training Context Length, Vocabulary Size, Language Focus | Pretraining scale, context regime, and linguistic coverage. |
| Input/output interface | Tokenizer Type, Tied Embeddings | Text input and output interface. |
| Sequence computation | Positional Encoding, Token Mixer | Position handling and cross-token computation. |
| Block transformation | Normalization, Non-linearity | Within-block transformation. |

## Exclusions

- **Release Date:** excluded because it is a temporal proxy for many unobserved design and training changes.
- **Depth / Width:** excluded because it is deterministically derived from retained properties.
- **Attention Type:** excluded because it is a broad proxy for retained sequence features.
- **FFN / Gating Type:** sensitivity-only because its RDM correlates `0.642736` with Token Mixer.

These features remain available in the preliminary full-feature analysis and can be reported as sensitivity analyses.

## Correlations between groups

No averaging, canonical correlation, or SVD is used.

1. Compute one model RDM for every metadata feature.
2. Vectorize the upper triangle of each RDM, giving 946 model-pair values.
3. For every pair of groups, compute the Pearson correlation between every member RDM in the first group and every member RDM in the second group.
4. Select the correlation with the largest absolute value while retaining its sign:

\[
r_{gh} = r_{f^*k^*}, \qquad
(f^*,k^*) = \arg\max_{f\in g,\,k\in h}|r_{fk}|.
\]

The heatmap therefore remains signed, with a symmetric color scale from `-1` to `1`. The largest cross-group correlation is `0.359823`; the within-block Normalization × Non-linearity correlation is `0.754977`.

These correlations are descriptive: model-pair observations sharing a model are dependent, so ordinary dyad-level correlation tests are inappropriate.

## Validated results

| Analysis | Fit, mean ± SD | Block-transformation FI, mean ± SD |
|---|---:|---:|
| RC MLEM | `0.519293 ± 0.006003` | `0.213721 ± 0.009070` |
| Long-range MLEM | `0.429991 ± 0.009705` | `0.037152 ± 0.002198` |
| RC RSA | `0.455502 ± 0.001803` | `0.008380 ± 0.000454` |
| Long-range RSA | `0.471260 ± 0.000769` | `0.009459 ± 0.000403` |

MLEM distances are model-level after DTW aggregation across layers. RSA distances are model-level sentence-RDM distances. Both use five folds, 100 grouped model-level permutations, nominal categorical distances, raw FI, and SD error bars.

The final cohort contains 38 retained base models and six RWKV-4 checkpoints. Falcon-Mamba, Gemma, and RWKV-7 are excluded. Canonical parquets are generated with `main.py` from:

```text
experiments/think_alike/families/config.yaml
experiments/think_alike/long_range_2/config.yaml
experiments/think_alike/rsa/model/config.yaml
experiments/think_alike/rsa/model_long_range_2/config.yaml
```

All figures are generated with `sh experiments/think_alike/families/main_results.sh`; each script reads the canonical parquets directly.
