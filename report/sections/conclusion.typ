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
$approx 2.1 times$ the variance of a single mean-direction logit lens while sharing almost none
of its top tokens, confirming that a one-shot per-expert readout under-reads a polysemantic
expert. We therefore read Expert Pursuit not as evidence that experts are crisp concept
detectors, but as a sparse, honest summary of the *limited* low-dimensional structure a single
expert carries --- consistent with the view that MoE semantics live largely in cross-layer
routing paths rather than individual experts @monosemanticpaths2026.

Finally, a causal toxic-expert circuit study --- with all identifiers fit on a train split and
all interventions scored on held-out prompts --- draws a sharp line between vocabulary
association and causation. The experts that causally drive toxic generation span all depths and
include strong suppressors, unlike the late-layer specialists that pursuit surfaces; a one-pass
gate attribution-patching score reproduces the expensive ablation grid moderately overall
($r approx 0.69$) and faithfully in the late layers ($approx 0.93$). But the headline causal
result is a *negative* one that sharpens the contribution: held out, single-expert top-$k$
knockout is near-inert for _every_ identifier --- causal scores order the experts correctly but
barely move toxicity --- because the behavior is spread redundantly across the active top-$k$
ensemble. This is the opposite of Head Pursuit's strongly-causal heads, so the SOMP _description_
transfers from heads to experts but the causal _localization_ does not. What does recover control
is acting on the shared residual direction: gentle project-out suppresses toxicity cleanly, while
aggressive additive steering works only by degrading generation indiscriminately. Correlational
specialization summaries are thus a starting point for hypotheses, not a substitute for a causal
test --- and in a redundant MoE, even causal _expert_ identification is better cashed out as a
direction than as an edit to a few experts.

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
