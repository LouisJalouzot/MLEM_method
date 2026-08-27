# Polynomial population simulation

## Purpose

This simulation separates two phenomena that have different meanings for MLEM:

- **factorial conjunctions** are context-dependent products of stimulus features;
- **cross-feature coupling** is shared tuning of population responses to otherwise additive feature contrasts.

MLEM off-diagonal coefficients can represent stable cross-feature coupling. A single global Mahalanobis matrix generally cannot represent factorial conjunctions because their local geometry depends on stimulus context.

## Generative procedure

Let $Z$ contain the encoded stimulus coordinates. The simulation standardizes these coordinates into centered contrasts $C$ before constructing pairwise products. It retains products between distinct semantic features and discards products between two encoding axes of the same categorical feature.

Centering makes conjunctions orthogonal to main effects in the population under the independent stimulus generator, but finite samples retain accidental correlations through sampling variability and categorical imbalance. Each conjunction column is therefore residualized against the intercept and all main-effect contrasts:

$$
Q_\perp = \left(I-P_{[\mathbf 1,C]}\right)Q.
$$

This makes the finite-population decomposition exact and prevents estimators from explaining nominal conjunction signal through leaked main effects. It is a benchmark-control step rather than a claim that residualization is required asymptotically. The retained main and conjunction columns are then standardized, giving

$$
H=[C,Q_\perp].
$$

Column standardization makes the sampled group strength, rather than arbitrary basis scale, determine each term's expected contribution to the response. This scaling was already part of the classical Poly simulation.

Each semantic main-effect or conjunction group is activated independently. Active groups receive a log-normal strength, divided by the square root of their number of encoded columns. With the canonical settings, $H$ has 132 columns grouped into 13 main-effect and 78 conjunction groups, for 91 groups total.

One random $d$-dimensional population-tuning direction is drawn for each column of $H$, normalized, and multiplied by its group scale. Coupling adds no columns to $H$. Instead, each pair of semantic features is selected independently with probability $p_{\mathrm{coupling}}$. An active pair draws a log-normal strength $s$, a unit response direction $v$, and unit contrasts $u_g$ and $u_h$ within its two feature blocks. The component

$$
A_g \mathrel{+}= \frac{s}{\sqrt{2}}u_gv^\top,
\qquad
A_h \mathrel{+}= \frac{s}{\sqrt{2}}u_hv^\top
$$

adds shared tuning with total component energy $s^2$. It creates the cross-feature block

$$
(AA^\top)_{gh}\supset \frac{s^2}{2}u_gu_h^\top.
$$

The sign and coordinate pattern are random through the sampled contrasts. Multiple active pairs may contribute independently to the same feature block. Conjunction tuning directions remain independently drawn.

Finally,

$$
Y_0=HA,\qquad Y=Y_0+\epsilon,
$$

where $\epsilon$ is correlated anisotropic Gaussian noise with optional sample-level outliers. A noise sweep rescales one fixed base realization for each replicate.

MLEM receives the original encoded $Z$. The oracle receives the frozen $H$. Thus, conjunctions deliberately create nonlinear misspecification for a global metric on $Z$, whereas coupling creates stable off-diagonal geometry that such a metric can represent.

## Literature grounding

The ingredients are standard, although this exact combination is a benchmark design rather than a canonical published generator.

| Component | Grounding | Status here |
|---|---|---|
| Main effects plus tensor-product conjunctions | Functional and tensor-product ANOVA decompose a response into a constant, main effects, and higher-order tensor-product terms (Lin, 2000). | Standard construction; finite-sample residualization is an explicit orthogonalization step. |
| Linear population encoding $Y_0=HA$ | Encoding models describe activity profiles through weighted features. Diedrichsen and Kriegeskorte (2017) place encoding, PCM, and RSA in a common second-moment framework. | Standard linear encoding model. |
| Random population tuning and covariance | PCM treats activity profiles as random variables, commonly Gaussian, whose second moment determines representational geometry. | Standard random-effects interpretation. |
| Nonlinear conjunctions | Diverse nonlinear mixed selectivity produces high-dimensional representations that support flexible linear readout (Rigotti et al., 2013; Fusi et al., 2016). | The pairwise conjunction bank is a simple controlled instance. |
| Pairwise shared-tuning components | Covariance of feature-tuning directions is compatible with the random-effects/second-moment framework. | Bernoulli-selected, rank-one pair components are a transparent benchmark construction, not a specific published neural circuit model. |
| Sparse log-normal group strengths, anisotropic noise, and outliers | These provide heterogeneous signal and robustness stress tests. | Pragmatic benchmark choices; no claim of a unique canonical form. |

## Recovery targets

The 78 conjunction groups in $H$ and the 78 cross-feature blocks of a raw-$Z$ metric share feature-pair names but represent different quantities. They must not be compared as one 91-group FI vector.

- Main-effect PFI recovery may compare the 13 main groups between an estimator and the oracle.
- Conjunction PFI is defined in $H$-space. A raw-$Z$ global metric has no corresponding conjunction input.
- Coupling recovery should compare estimated semantic off-diagonal $W$ blocks with the true linear-main metric $W_{\mathrm{main}}$, using block norms, signed coordinate alignment, or singular values.

## Interpretation and limitations

- `p_conjunction` is the probability that a semantic conjunction group is active.
- `p_coupling` is the probability that a semantic feature pair receives one shared-tuning component.
- Each coupling component has the same log-normal strength law and total-energy convention as the main and conjunction groups.
- Coordinate contrasts within categorical blocks are basis-dependent. Interpret coupling through semantic blocks, block norms, singular values, mapped contrasts, or grouped PFI.
- Sample residualization makes conjunctions exactly orthogonal to lower-order terms in the generated finite population. It does not make their geometry representable by one global $W$ on raw $Z$.
- The simulator should be described as **PCM/ANOVA-inspired** or as a **random-effects polynomial encoding benchmark**, rather than as a direct implementation of any single cited model.

## References

- Lin, Y. (2000). Tensor product space ANOVA models. *Annals of Statistics, 28*(3), 734–755. <https://doi.org/10.1214/aos/1015951996>
- Rigotti, M., et al. (2013). The importance of mixed selectivity in complex cognitive tasks. *Nature, 497*, 585–590. <https://doi.org/10.1038/nature12160>
- Fusi, S., Miller, E. K., & Rigotti, M. (2016). Why neurons mix: high dimensionality for higher cognition. *Current Opinion in Neurobiology, 37*, 66–74. <https://doi.org/10.1016/j.conb.2016.01.010>
- Diedrichsen, J., & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis. *PLoS Computational Biology, 13*(4), e1005508. <https://doi.org/10.1371/journal.pcbi.1005508>
