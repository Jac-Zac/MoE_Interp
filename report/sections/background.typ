= Background

== Transformer Residual Stream

A transformer @vaswani2017attention builds a $d$-dimensional residual stream across $L$ layers
@elhage2021mathematical. At the final layer, this stream is projected through the
unembedding matrix $bold(D) in RR^(v times d)$ to produce logits over the vocabulary.
Each row of $bold(D)$ is a direction associated with a specific token.

In Expert Pursuit, we row-normalize this dictionary before sparse coding so token matching
is driven by direction rather than vector magnitude. This is especially important because
the pursuit procedure is scale-sensitive.

== Mixture-of-Experts

In MoE models @shazeer2017outrageously @fedus2022switch, each FFN layer is replaced by
a set of expert FFNs and a learned router. For each token, the router selects the top-$k$
experts and assigns gating weights. The MoE output is:

$ "MoE"(bold(x)) = sum_(e in cal(T)_k (bold(x))) g_e (bold(x)) dot f_e (bold(x)) $ <eq:moe>

where $g_e$ is the gating weight and $f_e$ is the expert FFN output. The gated output
$g_e (bold(x)) dot f_e (bold(x))$ is the actual vector that expert $e$ adds to the
residual stream.

== Related MoE Interpretability Work

Recent work suggests several complementary views of MoE specialization. The domain/driver
expert framework @wang2025whatgets separates experts that specialize in semantic domains
from experts that exert strong causal influence on predictions @ternovtsii2026geometric. Semantic routing studies
@jin2025probing test whether input meaning influences routing patterns, while MoE editing
methods @moeedit2025 ask how to change expert behavior without perturbing routing. Newer
interpretability analyses such as MoE Lens @chaudhari2026moelens emphasize that a small set
of experts can dominate decoding behavior. These results motivate treating expert activations
as structured, interpretable objects rather than only efficiency mechanisms.

A second, more cautionary line of work bears directly on what a single-expert readout can
recover. MoE experts are *polysemantic*: they pack many unrelated features into superposition,
so one expert rarely corresponds to one human concept. @lecomte2025sparsity find experts are
somewhat *more* monosemantic than dense FFN neurons but still polysemantic; @monosemanticpaths2026
show that an individual expert at one layer yields little interpretable structure and that
semantics instead live in cross-layer routing *paths*; and @illusionspecialization2026 report
a domain-invariant ``standing committee'' of experts active across domains, i.e. specialization
is weaker than routing statistics suggest; @do2026domain likewise question whether
domain-specific experts exist at all, and @wang2026myth argue that expert assignment reflects
representation geometry rather than genuine semantic specialization. Architectures such as Monet @park2024monet and
intrinsically-interpretable MoEs @he2025intrinsic are built specifically to *force* the
monosemanticity that standard experts lack. The defensible reading is therefore not that
per-expert decomposition is uninformative, but that any *single-expert, single-readout* summary
under-reads a polysemantic expert --- a claim we test directly in @sec:results.

== SOMP

Simultaneous Orthogonal Matching Pursuit @tropp2006algorithms @mallat1993matching --- the
multi-sample generalization of Orthogonal Matching Pursuit @pati1993orthogonal --- is a
greedy sparse coding algorithm. Given a data matrix $bold(H) in RR^(n times d)$ (here,
aggregated expert activations across $n$ documents) and a dictionary
$bold(D) in RR^(v times d)$ (here, the unembedding matrix), SOMP iteratively selects
dictionary atoms that best explain the data:

$ bold(H) approx bold(W)^* bold(D) $ <eq:somp-approx>

At each step $t$, it selects the atom $p^t$ with the highest aggregate correlation across
all samples, adds it to the support set $SS^(t+1)$, and refits coefficients via
least-squares. After $T$ steps, we obtain a sparse set of vocabulary tokens that
characterize the expert's behavior. The explained variance ratio (EVR) at each step
measures how much signal the selected atoms capture.

This is a multi-sample generalization of the Logit Lens @nostalgebraist2020logitlens (and its
trained refinement, the Tuned Lens @belrose2023tunedlens): projecting a single activation onto
$bold(D)$ is equivalent to one step of Matching Pursuit on one sample. SOMP extends this by operating on many samples simultaneously and
selecting multiple atoms @basile2025headpursuit.
