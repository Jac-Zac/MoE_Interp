= Conclusion and Future Work

We presented Expert Pursuit, an adaptation of Head Pursuit to MoE expert FFNs. By running
SOMP on aggregated gated outputs against the unembedding dictionary, we obtain human-readable
token summaries for each expert. Applied to OLMoE-1B-7B-Instruct on 10,000 TriviaQA
questions, the method recovers interpretable specialists in numbers, geography, names,
biology, kinship, and entertainment, concentrated in the later layers. The concept-
restricted mode enables targeted queries that provide convergent evidence for candidate
specialization along specific semantic axes.

Two results temper the "specialist" reading and align with recent MoE interpretability work.
First, only a minority of experts have clean token summaries: the median final EVR is low, so a
small vocabulary basis explains little output variance. This is compatible with superposition
and polysemanticity @lecomte2025sparsity @illusionspecialization2026, but does not establish either
without a calibrated null. Second, even
atom-for-atom SOMP explains $approx 2 times$ the variance of a single mean-direction logit lens
(and $approx 14 times$ when its 10-atom basis is compared with the lens's single direction),
showing that the one-shot mean-direction readout captures less variance. We therefore read Expert Pursuit not as evidence that experts are crisp concept
detectors, but as a sparse, honest summary of the *limited* low-dimensional structure a single
expert carries --- consistent with the view that MoE semantics live largely in cross-layer
routing paths rather than individual experts @monosemanticpaths2026.

Finally, the gate intervention study separates vocabulary association from the effect of a specific
post-top-$k$ gate ablation. Gate-AtP has the largest knockout point estimate on all three prompt
sets, and its correlation with exhaustive gate ablation is $r approx 0.69$ pooled and $approx 0.93$
over layers 12--15. The `countries` and `numbers` results are exploratory ($n = 18$ each, separately
authored prompt sets, one random draw); the larger held-out RealToxicityPrompts experiment shows
only a weak sparse handle for the offensive-word probe. Because remaining weights are not
renormalized and replacement experts are not selected, robustness to knockout should not be called
router compensation without an additional experiment. The probe is also lexical rather than an
independent toxicity classifier. Correlational summaries are therefore useful hypothesis generators,
not substitutes for controlled causal and behavioral evaluation.

== Limitations

- *Position and routing support.* TriviaQA is read only at the shared final template token, so the
  descriptive sample covers 334 of 1,024 layer--expert slots. Pile-10k supplies the all-slot EVR
  comparison but answers a different, completion-style question. Content-token averaging remains
  the most important extraction ablation.
- *Readout approximation.* Earlier expert outputs are scaled with the final residual's RMSNorm
  factor but are not propagated through later layers. The reconstruction has unit tests against
  the expert equations, not an end-to-end real-model error measurement against the actual block
  and final residual outputs.
- *Dictionary dependence.* EVR and atom-family coherence depend on the normalized unembedding,
  the number of selected atoms, and a fixed cosine threshold. A random-dictionary floor,
  frequency-matched token null, PCA ceiling, and threshold sensitivity analysis are missing.
- *Causal evaluation.* The `countries` and `numbers` prompt sets are small and incompletely
  archived; each comparison uses one random set. The rollout propensity follows a different
  generated context under each intervention, and the offensive lexicon is not an independent
  toxicity measure. Distinct-1 detects repetition but not factuality or general fluency.
- *Scope.* OLMoE is the primary model. The smaller GPT-OSS check supports dictionary-alignment and
  gate-AtP correlation observations, not a powered cross-model intervention result.

== Future Work

The following close the gap to a paper-strength causal claim, roughly in order of importance.

- *Break the metric circularity.* The offensive-token logit probe is built from the same word
  list that defines the concept it grades, so the knockout is partly judged by the thing it
  removes. Scoring the held-out continuations with an *independent* toxicity
  classifier (Detoxify @hanu2020detoxify or Perspective API) as the headline metric, keeping the
  logit probe as a cheap proxy, is the single largest credibility upgrade and pairs naturally
  with the existing held-out split.

- *Calibrate the EVR floor.* A random-dictionary / PCA ceiling would calibrate the EVR result
  against the floor that $k$ free atoms give. The cumulative and co-firing ablations in
  @sec:causal establish robustness to the tested interventions but do not isolate whether it comes
  from concurrent experts, changed later routing, or distributed non-expert computation.

- *Cross-layer paths.* Since semantics in MoEs appear to live in routing trajectories rather than
  single experts @monosemanticpaths2026, and even the tested co-firing _groups_ have modest effects
  under post-top-$k$ gate downweighting, decomposing and intervening *along an expert path* (the
  sequence of experts a token routes through) is the natural next test of distributed mechanisms.

- *Scale the second study.* The GPT-OSS run replicates the descriptive and faithfulness claims
  but identified experts on only $n_"train" = 50$ prompts; re-running the full held-out circuit at
  OLMoE scale, adding a third model (Mixtral @jiang2024mixtral, DeepSeek-MoE @dai2024deepseekmoe)
  and more concepts, would turn the replication into a genuine cross-architecture result.
