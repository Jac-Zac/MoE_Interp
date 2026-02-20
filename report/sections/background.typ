= Background

== Transformer Residual Stream

A transformer builds a $d$-dimensional residual stream across $L$ layers
@elhage2021mathematical. At the final layer, this stream is projected through the
unembedding matrix $bold(D) in RR^(v times d)$ to produce logits over the vocabulary.
Each row of $bold(D)$ is a direction associated with a specific token.

== Mixture-of-Experts

In MoE models @shazeer2017outrageously @fedus2022switch, each FFN layer is replaced by
a set of expert FFNs and a learned router. For each token, the router selects the top-$k$
experts and assigns gating weights. The MoE output is:

$ "MoE"(bold(x)) = sum_(e in cal(T)_k (bold(x))) g_e (bold(x)) dot f_e (bold(x)) $ <eq:moe>

where $g_e$ is the gating weight and $f_e$ is the expert FFN output. The gated output
$g_e (bold(x)) dot f_e (bold(x))$ is the actual vector that expert $e$ adds to the
residual stream.

== SOMP

Simultaneous Orthogonal Matching Pursuit @tropp2006algorithms @mallat1993matching is a
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

This is a multi-sample generalization of the Logit Lens @nostalgebraist2020logitlens:
projecting a single activation onto $bold(D)$ is equivalent to one step of Matching
Pursuit on one sample. SOMP extends this by operating on many samples simultaneously and
selecting multiple atoms @basile2025headpursuit.
