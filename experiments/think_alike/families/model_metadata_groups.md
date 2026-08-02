# Model metadata groups

## Main-analysis groups

| Group | Features | Motivation |
|---|---|---|
| Training Procedure | Weight Provenance, Distillation Objective, Learning Rate Schedule, Warmup Fraction, Training Precision | Describes how model weights were initialized and optimized during pretraining. |
| Architecture Type | Architecture, Normalization, Non-linearity, Positional Encoding, Attention Type, Tied Embeddings | Describes concrete architectural choices without using the aggregate Family label. |
| Model Scale and Shape | Num. Parameters, Width, Depth | Groups strongly related measures of model capacity and shape. |
| Pretraining Data | Training Tokens, Vocabulary Size, Language Focus | Describes the scale and linguistic scope of pretraining data. |

The new training properties use the following coding rules:

- **Weight Provenance** describes the weights used to begin the reported pretraining phase: `From scratch`, `Pruned teacher`, or `Converted predecessor`.
- **Distillation Objective** is `True` only when pretraining directly used teacher outputs as targets. Teacher-generated synthetic text remains a property of the training data and does not count as direct distillation.

The pruned and distilled models are Llama 3.2 and Ministral-3. The RWKV-7 checkpoints were converted from and continually pretrained from earlier RWKV checkpoints. The other checkpoints were trained from scratch. Sources: the [Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md), the [Ministral-3 report](https://arxiv.org/abs/2601.08584), and the [RWKV-7 report](https://arxiv.org/abs/2503.14456).

## Additional training properties in the preliminary analysis

We retain properties documented for more than half of the 43 models:

| Property | Definition | Models | Coverage |
|---|---|---:|---:|
| Checkpoint Averaging | Whether the released weights average multiple independently trained branches/checkpoints instead of selecting one checkpoint | 22 | 51% |
| Learning Rate Schedule | Cosine, polynomial, or cosine followed by linear decay | 22 | 51% |
| Warmup Fraction | Fraction of training updates or tokens used for learning-rate warmup | 22 | 51% |
| Training Sequence Length | Maximum sequence length used during base pretraining, log2-transformed | 35 | 81% |
| Staged Training | Whether base pretraining contains distinct training stages | 35 | 81% |
| Long-context Extension | Whether a later base-pretraining stage explicitly increases sequence length | 35 | 81% |
| Training Precision | Numerical precision used during base pretraining | 23 | 53% |

Checkpoint averaging, or model souping, averages the parameters of several trained checkpoints or branches to produce the released model. OLMo-2 7B and 13B average several independently annealed stage-2 branches, while OLMo-2 1B selects a single branch.

Missing values do not form an `Unknown` category. For the preliminary MLEM, a feature contributes zero distance to model pairs for which either value is missing. Feature-RDM correlations use only model pairs observed for both features. This avoids making reporting practices directly predictive, although the different correlations are based on different subsets of models.

The largest absolute correlation of each additional property is:

| Property | Correlation | Other property |
|---|---:|---|
| Checkpoint Averaging | `.746` | Staged Training |
| Learning Rate Schedule | `.518` | Staged Training |
| Warmup Fraction | `.596` | Depth / Width |
| Training Sequence Length | `.872` | Long-context Extension |
| Staged Training | `.746` | Checkpoint Averaging |
| Long-context Extension | `1.000` | Attention Type |
| Training Precision | `.618` | Release Date |

We add Learning Rate Schedule, Warmup Fraction, and Training Precision to the main Training Procedure group. Their largest cross-group correlation is `.517`, between Learning Rate Schedule and Vocabulary Size. Checkpoint Averaging, Training Sequence Length, Staged Training, and Long-context Extension remain preliminary because their correlations with other groups range from `.702` to `1.000`. Exact optimizer, global batch size, weight decay, and gradient clipping remain below 50% coverage.

The preliminary MLEM results are:

| Additional property | Relative Clause FI | Long Range Agreement FI |
|---|---:|---:|
| Checkpoint Averaging | `-.000 ± <.001` | `-.000 ± <.001` |
| Learning Rate Schedule | `.042 ± .002` | `.000 ± <.001` |
| Warmup Fraction | `.002 ± <.001` | `-.000 ± <.001` |
| Training Sequence Length | `.000 ± <.001` | `-.000 ± <.001` |
| Staged Training | `.047 ± .001` | `-.000 ± <.001` |
| Long-context Extension | `.017 ± .002` | `-.000 ± <.001` |
| Training Precision | `.000 ± <.001` | `.000 ± <.001` |

The full preliminary fit is `.610 ± .005` for Relative Clause DTW and `.521 ± .011` for Long Range Agreement DTW. The added properties improve the Relative Clause fit, mainly through Staged Training and Learning Rate Schedule. They have little unique importance for Long Range Agreement once the other metadata properties are included.

Sources: the [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [OPT report](https://arxiv.org/abs/2205.01068), [Pythia configurations](https://github.com/EleutherAI/pythia/tree/main/models), [OLMo configurations](https://github.com/allenai/OLMo), [Mamba training clarification](https://github.com/state-spaces/mamba/issues/184), [Ministral-3 report](https://arxiv.org/abs/2601.08584), [Qwen3 report](https://arxiv.org/abs/2505.09388), and [RWKV-7 report](https://arxiv.org/abs/2503.14456).

## Exclusions

- **Family:** excluded because it aggregates the concrete architecture, training, and data properties that the analysis aims to interpret.
- **Release Date:** excluded because it is a temporal proxy for many unobserved design and training changes.
- **Depth / Width:** excluded because it is deterministically derived from retained properties.
- **Tokenizer Type:** excluded because it is almost constant in this sample and its only major contrast is already captured by retained architecture and data properties.
- **FFN / Gating Type:** excluded because it is redundant with retained architecture properties and raises the largest cross-group correlation above `.5`.

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

| Groups | Largest correlation |
|---|---:|
| Training Procedure / Architecture Type | `.479` |
| Training Procedure / Model Scale and Shape | `.218` |
| Training Procedure / Pretraining Data | `.517` |
| Architecture Type / Model Scale and Shape | `.290` |
| Architecture Type / Pretraining Data | `.485` |
| Model Scale and Shape / Pretraining Data | `.091` |

The largest absolute cross-group correlation is `.517`, between Learning Rate Schedule and Vocabulary Size. These correlations are descriptive: model-pair observations sharing a model are dependent, so ordinary dyad-level correlation tests are inappropriate.

## Model-level results

| Target | Spearman, mean ± SD | Training Procedure | Architecture Type | Model Scale and Shape | Pretraining Data |
|---|---:|---:|---:|---:|---:|
| Relative Clause DTW | `.512 ± .007` | `.064 ± .002` | `.130 ± .007` | `.076 ± .002` | `.163 ± .003` |
| Long Range Agreement DTW | `.474 ± .013` | `.162 ± .004` | `.189 ± .005` | `.120 ± .007` | `.001 ± <.001` |
| Raw model-level RSA | `.414` | `-.003` | `.114` | `.046` | `.220` |

Pretraining Data leads for Relative Clause DTW and raw RSA, while Architecture Type leads for Long Range Agreement DTW. Grouped Feature Importance applies the same model-label permutation to every member RDM in a group and measures the decrease in Spearman correlation.
