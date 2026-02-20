= Introduction

Mixture-of-Experts (MoE) language models route each token to a small subset of experts,
enabling large capacity with sparse computation @shazeer2017outrageously @fedus2022switch.
While routing statistics reveal which experts are used, they do not explain what each
expert writes into the residual stream.

We adapt Head Pursuit @basile2025headpursuit to MoE experts. For each expert and layer, we
analyze its gated output --- the expert FFN output scaled by the router weight --- and
decompose it into a sparse set of vocabulary directions using the model's unembedding
matrix as a dictionary. The resulting atoms are human-readable tokens that summarize
expert behavior.

Following Head Pursuit, we use TriviaQA @joshi2017triviaqa questions as documents and
average expert gated outputs over routed question-content tokens, excluding chat template
markers. This mirrors the aggregation strategy used for attention heads.

Our contributions:
- A residual-stream decomposition for MoE experts based on gated outputs.
- Application of SOMP with the unembedding dictionary to obtain sparse, token-level
  expert summaries.
- A TriviaQA-based pipeline for analyzing expert specialization in OLMoE.
