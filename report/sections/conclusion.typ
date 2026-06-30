= Conclusion and Future Work

We presented Expert Pursuit, an adaptation of Head Pursuit to MoE expert FFNs. By running
SOMP on aggregated gated outputs against the unembedding dictionary, we obtain human-readable
token summaries for each expert. Applied to OLMoE-1B-7B-Instruct on 10,000 TriviaQA
questions, the method recovers interpretable specialists in numbers, geography, names,
biology, kinship, and entertainment, concentrated in the later layers. The concept-
restricted mode enables targeted queries that confirm and quantify specialization along
specific semantic axes.

Two results temper the "specialist" reading and align with recent MoE interpretability work.
First, only a minority of experts are cleanly specialized: the median final EVR is low, so most
experts are polysemantic @lecomte2025sparsity @illusionspecialization2026. Second, even
atom-for-atom SOMP explains $approx 2 times$ the variance of a single mean-direction logit lens
(and $approx 14 times$ with its full 10-atom basis) while sharing almost none of its top tokens,
confirming that a one-shot per-expert readout under-reads a polysemantic expert. We therefore read Expert Pursuit not as evidence that experts are crisp concept
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
redundant. The SOMP _description_ transfers from heads to experts, but the causal _localization_
does not --- and this holds even under Head Pursuit's own $alpha = -1$ rescale intervention, which
on our experts ties a layer-matched random control (@tab:hp): the obstruction is $8$-of-$64$
routing redundancy, not the choice of intervention. Second, the correlational
SOMP selector is never cleanly causal --- where it lowers a metric it does so only by collapsing
generation into garbage (distinct-1 $0.27$--$0.59$), a failure a coherence guard exposes
immediately. A cheap one-pass gate gradient recovers the causal influence faithfully
($r approx 0.69$ pooled, $approx 0.93$ in the late layers where the controllable signal lives), so
the expensive patching grid is needed only to validate it. The diffuse tail (toxicity) is the
honest endpoint: it has no usable expert handle at all --- knockout is inert and even expert-output
steering of the causal set only suppresses it weakly and non-specifically --- because the behavior
is semantic and fully distributed rather than carried by any sparse expert set. Correlational
specialization summaries are thus a starting point for hypotheses, not a substitute for a causal,
coherence-aware test. In a redundant MoE, causal _influence_ over a concept's experts is
recoverable for the localizable concepts, but expert _necessity_ is an illusion of redundant
routing, and the least lexical behaviors are not expert-localizable at all.

== Future Work

The descriptive and faithfulness results are solid; the following close the gap to a
paper-strength _causal_ claim, roughly in order of importance.

- *Break the metric circularity.* The toxic-logit probe is built from the same offensive word
  list as the diff-of-means steering direction it grades, so the intervention is partly judged
  by the thing it removes. Scoring the held-out continuations with an *independent* toxicity
  classifier (Detoxify @hanu2020detoxify or Perspective API) as the headline metric, keeping the
  logit probe as a cheap proxy, is the single largest credibility upgrade and pairs naturally
  with the existing held-out split.

- *Calibrate the EVR floor.* A random-dictionary / PCA ceiling would calibrate the EVR result
  against the floor that $k$ free atoms give (the sufficiency curve and co-firing group ablation
  in @sec:causal already confirm the top-$k$ redundancy account, and bootstrap CIs resolve the
  one significant knockout effect).

- *Cross-layer paths.* Since semantics in MoEs appear to live in routing trajectories rather than
  single experts @monosemanticpaths2026, and even co-firing _groups_ are knockout-redundant,
  decomposing and intervening *along an expert path* (the sequence of experts a token routes
  through) is the natural next fix for the redundancy that defeats sparse knockout.

- *Scale the second study.* The GPT-OSS run replicates the descriptive and faithfulness claims
  but identified experts on only $n_"train" = 50$ prompts; re-running the full held-out circuit at
  OLMoE scale, adding a third model (Mixtral @jiang2024mixtral, DeepSeek-MoE @dai2024deepseekmoe)
  and more concepts, would turn the replication into a genuine cross-architecture result.
