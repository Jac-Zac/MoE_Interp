= Methods

We adapt Head Pursuit @basile2025headpursuit from attention heads to MoE experts. Where
Head Pursuit decomposes per-head residual stream contributions, we decompose per-expert gated
outputs using the same SOMP-based sparse coding framework.

== Dataset

We use TriviaQA @joshi2017triviaqa (RC configuration, train split), following the Head
Pursuit setup. Each question is one document. Questions are wrapped in the model's chat
template without any additional QA prompt --- only the raw question text is presented to the
model.

== Model and Activation Extraction

We target OLMoE-1B-7B-Instruct @muennighoff2024olmoe: 16 layers, 64 experts per layer,
top-8 routing, $d = 2048$. Using `nnsight` for model tracing, we capture for each token: (1)
the router's top-$k$ expert indices and gating weights, and (2) the raw FFN output
$f_e (bold(x)_i)$ from each selected expert.

To keep the capture aligned with the model's positional encoding, prompts are traced in
right-padded batches and we extract the last real token from each prompt. The gated expert
output is then multiplied by the router weight and passed through an approximate final
RMSNorm using the residual-stream second moment, matching the scale of the model's final
representation.

== Aggregation

For each expert $e$ at layer $l$, we compute the gated output at the last token position for
each document $j$:

$ bold(e)_(e,l)^j = g_e (bold(x)_j) dot f_e (bold(x)_j) $ <eq:expert-agg>

Stacking across $n$ documents yields $macron(bold(E))_(e,l) in RR^(n times d)$, the input to
SOMP. Documents where expert $e$ receives no routed tokens are excluded.

== SOMP Decomposition

For each expert, we run SOMP with the L2-normalized unembedding matrix as dictionary and
$T = 50$ iterations. This produces a ranked list of vocabulary tokens that best explain the
expert's variance across questions, along with cumulative EVR scores.

== Word-Dictionary Mode

In addition to the base unembedding dictionary, the pipeline can augment the dictionary with
averaged atoms for multi-token words. These merged atoms are re-normalized after averaging so
their direction, rather than their smaller norm, determines their influence during SOMP.

== Two Modes of Analysis

The pipeline supports two complementary analysis modes.

*Full-dictionary mode.* SOMP searches the entire vocabulary ($v approx 50{,}000$ tokens). The
output is an unrestricted ranked list of tokens that summarize the expert's aggregate
behavior --- analogous to a per-expert logit lens applied across many documents. This is the
primary mode for discovering what each expert specializes in.

*Concept-restricted mode.* The dictionary is restricted to the token IDs corresponding to a
predefined concept word list (e.g., `numbers`, `countries`). SOMP then decomposes each
expert's activations against only those directions, and final EVR scores rank experts by how
strongly they respond to that concept. This allows targeted queries such as: _which experts
are most active on numeric content?_ The concept word lists are defined in `src/concepts.py`.

The two modes are complementary: full-dictionary pursuit discovers specialists without prior
hypotheses, while concept-restricted pursuit quantifies specialization along a specific
semantic axis and is directly actionable for targeted interventions such as activation
steering.
