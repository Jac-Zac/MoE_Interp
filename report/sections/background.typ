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

Recent work treats the *expert* --- not the neuron --- as the natural unit for interpreting
MoEs. @wang2025whatgets distinguish *domain* experts (specialized to a topic) from *driver*
experts (large causal effect on the output), and @chaudhari2026moelens show a single
top-weighted expert can approximate a whole layer's contribution to decoding. @herbst2026expert
argue the expert is a more effective interpretable unit than the neuron, automatically
describing hundreds of experts. This view is *actionable*: @ternovtsii2026geometric control
behavior by steering routing through its geometry, and @do2026domain find domain-specific
experts do exist and edit their weights for training-free domain control. Together this
motivates treating expert outputs as structured, interpretable, and *intervenable* objects ---
the premise of our circuit study (@sec:causal).

A second, cautionary line bounds what any *single-expert, single-readout* summary can recover.
MoE experts are *polysemantic*, packing unrelated features into superposition, so one expert
rarely maps to one concept. @lecomte2025sparsity and @herbst2026expert both find experts
somewhat *more* monosemantic than dense FFN neurons but still far from one-concept-per-expert,
with monosemanticity rising only as routing grows sparser. @monosemanticpaths2026 show an
individual expert yields little interpretable structure while cross-layer routing *paths* are
monosemantic; @illusionspecialization2026 find a domain-invariant ``standing committee'' that
carries most routing mass across domains; and @wang2026myth argue expert assignment reflects
representation *geometry* rather than genuine domain expertise. Architectures such as Monet
@park2024monet and intrinsically-interpretable MoEs @he2025intrinsic instead *force* the
monosemanticity standard experts lack. The defensible reading is not that per-expert
decomposition is uninformative, but that a single-readout summary under-reads a polysemantic
expert --- which we test directly in @sec:results.

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
