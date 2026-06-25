= Conclusion and Future Work

We presented Expert Pursuit, an adaptation of Head Pursuit to MoE expert FFNs. By running
SOMP on aggregated gated outputs against the unembedding dictionary, we obtain human-readable
token summaries for each expert. Applied to OLMoE-1B-7B-Instruct on 50,000 TriviaQA
questions, the method recovers interpretable specialists in numbers, geography, names,
biology, kinship, and entertainment, concentrated in the later layers. The concept-
restricted mode enables targeted queries that confirm and quantify specialization along
specific semantic axes.

Two results temper the "specialist" reading and align with recent MoE interpretability work.
First, only a minority of experts are cleanly specialized: the median final EVR is low, so most
experts are polysemantic @lecomte2025sparsity @illusionspecialization2026. Second, SOMP explains
$approx 2.6 times$ the variance of a single mean-direction logit lens while sharing almost none
of its top tokens, confirming that a one-shot per-expert readout under-reads a polysemantic
expert. We therefore read Expert Pursuit not as evidence that experts are crisp concept
detectors, but as a sparse, honest summary of the *limited* low-dimensional structure a single
expert carries --- consistent with the view that MoE semantics live largely in cross-layer
routing paths rather than individual experts @monosemanticpaths2026.

Finally, a causal toxic-expert circuit study draws a sharp line between vocabulary association
and causation. The experts that causally drive toxic generation span all depths and include
strong suppressors, unlike the late-layer specialists that pursuit and direct logit attribution
surface; a one-pass gate attribution-patching score reproduces the expensive ablation grid
faithfully ($r approx 0.80$) while the correlational token-association ranking does not. At
intervention time only the causally-identified experts matter --- knocking out the
SOMP-identified set is inert --- and the cleanest suppression comes from projecting the toxic
direction out of the residual stream. Correlational specialization summaries are thus a starting point for hypotheses,
not a substitute for a causal test.

== Future Work

- *Content-token averaging.* The current pipeline aggregates over the last token only.
  Averaging over question-content tokens (as in the original Head Pursuit) remains a possible
  extension if we want a closer apples-to-apples comparison with the original framework.

- *Scaling and coverage.* Many experts have fewer than 100 activations even at 50,000
  documents, particularly in early layers. Larger corpora or targeted prompting strategies
  would improve coverage.

- *Cross-model comparison.* Applying the same pipeline to other MoE models (Mixtral
  @jiang2024mixtral, DeepSeek-MoE @dai2024deepseekmoe, gpt-oss @openai2025gptoss) would test
  whether the observed specialization patterns are model-specific or a general property of MoE
  routing.

- *Cross-layer paths.* Since semantics in MoEs appear to live in routing trajectories rather
  than single experts @monosemanticpaths2026, decomposing the gated outputs *along an expert
  path* (a sequence of experts across layers selected for the same token), rather than one
  expert in isolation, is a natural way to extend pursuit toward the unit the literature
  suggests is monosemantic.
