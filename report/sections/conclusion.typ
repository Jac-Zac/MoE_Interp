= Conclusion and Future Work

We presented Expert Pursuit, an adaptation of Head Pursuit to MoE expert FFNs. By running
SOMP on aggregated gated outputs against the unembedding dictionary, we obtain human-readable
token summaries for each expert. Applied to OLMoE-1B-7B-Instruct on 50,000 TriviaQA
questions, the method recovers interpretable specialists in numbers, geography, names,
biology, religion, and entertainment, concentrated in the later layers. The concept-
restricted mode enables targeted queries that confirm and quantify specialization along
specific semantic axes.

== Future Work

- *Content-token averaging.* The current pipeline aggregates over the last token only.
  Averaging over question-content tokens (as in the original Head Pursuit) remains a possible
  extension if we want a closer apples-to-apples comparison with the original framework.

- *Activation steering.* The ranked expert list produced by concept-restricted pursuit
  directly identifies which experts to suppress or amplify to steer model behavior toward
  or away from a given concept. This connection is not yet implemented but is a natural
  next step.

- *Scaling and coverage.* Many experts have fewer than 100 activations even at 50,000
  documents, particularly in early layers. Larger corpora or targeted prompting strategies
  would improve coverage.

- *Cross-model comparison.* Applying the same pipeline to other MoE models (Mixtral,
  DeepSeek-MoE, gpt-oss) would test whether the observed specialization patterns are
  model-specific or a general property of MoE routing.
