= Methods

We adapt Head Pursuit @basile2025headpursuit from attention heads to MoE experts. Where
Head Pursuit decomposes per-head residual stream contributions, we decompose per-expert
gated outputs using the same SOMP-based sparse coding framework.

== Dataset

We use TriviaQA @joshi2017triviaqa (RC configuration, train split), following the Head
Pursuit setup. Each question is one document. Questions are wrapped in the model's chat
template without any additional QA prompt --- only the raw question text is presented to
the model. We average exclusively over question-content tokens, excluding chat template
markers (e.g., `<|user|>`, `<|assistant|>`).

== Model and Activation Extraction

We target OLMoE-1B-7B-Instruct @muennighoff2024olmoe: 16 layers, 64 experts per layer,
top-8 routing, $d = 2048$. Using `nnsight` for model tracing, we capture for each token:
(1) the router's top-$k$ expert indices and gating weights, and (2) the raw FFN output
$f_e (bold(x)_i)$ from each selected expert.

== Aggregation

For each expert $e$ at layer $l$, we compute the gated output per token and average over
the question-content tokens routed to that expert within a document $j$:

$ macron(bold(E))_(e,l)^j = frac(1, |cal(R)_(e,l)^j|) sum_(i in cal(R)_(e,l)^j) g_e (bold(x)_i) dot f_e (bold(x)_i) $ <eq:expert-agg>

Stacking across $n$ documents yields $macron(bold(E))_(e,l) in RR^(n times d)$, the
input to SOMP. Documents where expert $e$ receives no routed tokens are excluded.

== SOMP Decomposition

For each expert, we run SOMP with the L2-normalized unembedding matrix as dictionary and
$T = 50$ iterations. This produces a ranked list of vocabulary tokens that best explain
the expert's variance across questions, along with cumulative EVR scores. To probe a
specific concept (e.g., countries), we restrict the dictionary to the relevant token
subset and recompute SOMP.
