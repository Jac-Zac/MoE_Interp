= Introduction

Mixture-of-Experts (MoE) language models route each token to a small subset of experts,
enabling large capacity with sparse computation @shazeer2017outrageously @fedus2022switch.
Routing statistics show which experts are used, but not what each expert writes into the
residual stream.

We adapt Head Pursuit @basile2025headpursuit to MoE experts. For each expert and layer, we
analyze its gated output --- the expert FFN output scaled by the router weight --- and
decompose it into a sparse set of vocabulary directions using the model's unembedding
matrix as a dictionary. The resulting atoms are human-readable tokens that summarize expert
behavior.

Unlike the original Head Pursuit aggregation over question-content tokens, the current
implementation traces the last real token of each prompt and uses right-padding so token
positions remain aligned during batched tracing. This preserves the original motivation while
matching the implemented capture code.

We use TriviaQA @joshi2017triviaqa questions as documents and analyze one prompt-level
activation summary per document. The captured expert outputs are gated and normalized before
pursuit, and the unembedding dictionary is L2-normalized row-wise so SOMP operates on token
directions rather than raw embedding scale.

Our contributions:
- A residual-stream decomposition for MoE experts based on gated outputs.
- Application of SOMP with the unembedding dictionary to obtain sparse, token-level
  expert summaries.
- A TriviaQA-based pipeline for analyzing expert specialization in OLMoE.
