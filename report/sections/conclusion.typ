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

Finally, a causal circuit study across three concepts --- with all selectors fit on a train split
and all interventions scored on held-out prompts --- draws a sharp line between vocabulary
association and causation, and reveals that causal _controllability_ is a *gradient*: `countries`
is sharply localizable, `numbers` only weakly, toxicity not at all. Two findings recur along it.
First, the separating signal is *influence, not necessity*: localized steering of the gate-AtP
experts removes a localizable concept cleanly (country word-fraction $0.60 -> 0.03$ with coherence
intact) and does so _specifically_ (the other concept survives), yet knocking the same experts out
--- even the top $10%$ of all experts --- never removes the concept, because top-$k$ routing is
redundant. This is the opposite of Head Pursuit's strongly-causal heads: the SOMP _description_
transfers from heads to experts but the causal _localization_ does not. Second, the correlational
SOMP selector is never cleanly causal --- where it lowers a metric it does so only by collapsing
generation into garbage (distinct-1 $0.27$--$0.59$), a failure a coherence guard exposes
immediately. A cheap one-pass gate gradient recovers the causal influence faithfully
($r approx 0.69$ pooled, $approx 0.93$ in the late layers where the controllable signal lives), so
the expensive patching grid is needed only to validate it. The diffuse tail (toxicity) is the
honest endpoint: it has no usable expert handle at all --- knockout is inert and even expert-output
steering of the causal set only suppresses it weakly and non-specifically --- because the behavior
is semantic and fully distributed rather than carried by any sparse expert set. Correlational
specialization summaries are thus a starting point for hypotheses, not a substitute for a causal,
coherence-aware test --- and in a redundant MoE, causal _influence_ over a concept's experts is
recoverable for the localizable concepts but expert _necessity_ is an illusion of redundant
routing, with the least lexical behaviors not expert-localizable at all.

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
