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

We use TriviaQA @joshi2017triviaqa questions as documents, summarizing each by the gated,
normalized expert output at its last token. Unlike the original Head Pursuit aggregation over
question-content tokens, this single-token readout keeps batched tracing positionally aligned;
the capture and dictionary-normalization details are deferred to @sec:methods.

Our contributions:
- A residual-stream decomposition for MoE experts based on gated outputs.
- Application of SOMP with the unembedding dictionary to obtain sparse, token-level
  expert summaries.
- A TriviaQA-based pipeline for analyzing expert specialization in OLMoE.
- A held-out causal circuit study (activation patching, gate-AtP, project-out) showing that
  toxic behavior is spread redundantly across the active top-$k$ experts: single-expert knockout
  is near-inert, and control is recovered at the level of the shared residual direction rather
  than by editing individual experts.
