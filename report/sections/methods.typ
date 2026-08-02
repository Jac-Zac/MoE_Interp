= Methods <sec:methods>

We adapt Head Pursuit @basile2025headpursuit from attention heads to MoE experts. Where
Head Pursuit decomposes per-head residual stream contributions, we decompose per-expert gated
outputs using the same SOMP-based sparse coding framework.

== Dataset

We use TriviaQA @joshi2017triviaqa (`rc.nocontext`, train split, $n = 10,000$), following
the Head Pursuit setup. Each question is one document. The question is inserted into the fixed
instruction ``Answer the following question in 1--3 words only ... Question: {question} Answer:``
and then wrapped in the model's chat template with a generation prompt. Consequently, every
TriviaQA example has the same final template token; @sec:results discusses the resulting routing
concentration and uses pile-10k for all-expert aggregate comparisons.

== Model and Activation Extraction

We target OLMoE-1B-7B-Instruct @muennighoff2024olmoe: 16 layers, 64 experts per layer,
top-8 routing, $d = 2048$. Using `nnsight`, we tap each fused MoE block's boundary tuple:
hidden states, top-$k$ expert indices, and routing weights. The fused kernel does not expose
individual FFN outputs, so after the trace we reconstruct each selected expert's raw output
$f_e (bold(x)_i)$ from the saved hidden state and that expert's weights. This reconstruction
mirrors the OLMoE SwiGLU forward pass.

To keep the capture aligned with the model's positional encoding, prompts are traced in
right-padded batches and we extract the last real token from each prompt. The gated expert
output is multiplied by the router weight and by the final RMSNorm's shared, input-dependent
scale, computed from the final pre-norm residual's second moment. This is a *direct-effect
readout*: it places an earlier component on the final unembedding scale without propagating it
through later layers or accounting for how changing that component would alter the normalization
denominator.

== Aggregation

For each expert $e$ at layer $l$, we compute the scaled gated output at the last token position
for each document $j$:

$ bold(e)_(e,l)^j = bold(gamma) times frac(g_e (bold(x)_j) dot f_e (bold(x)_j),
sqrt(d^(-1) norm(bold(r)_j)_2^2 + epsilon)) $ <eq:expert-agg>

Here $bold(r)_j$ is the actual final pre-norm residual for that document, $bold(gamma)$ is the
final RMSNorm weight, and multiplication by $bold(gamma)$ is element-wise. The denominator is
held fixed while scaling the component.

Stacking across $n$ documents yields $macron(bold(E))_(e,l) in RR^(n times d)$, the input to
SOMP. Documents where expert $e$ receives no routed tokens are excluded.

== SOMP Decomposition

For each expert, we run SOMP (@app:somp-alg) with the L2-normalized unembedding matrix as
dictionary and $T = 25$ iterations. This produces a ranked list of vocabulary tokens that best
explain the expert's variance across questions, along with cumulative EVR scores.

== Analysis Modes

The pipeline supports two complementary analysis modes, and either can run over a
word-augmented dictionary in which multi-token words are appended as averaged atoms on top of
the base vocabulary and re-normalized, so their direction --- not their smaller norm ---
determines their influence during SOMP.

*Full-dictionary mode.* SOMP searches the entire vocabulary ($v approx$ 50,000 tokens). The
output is an unrestricted ranked list of tokens that summarize the expert's aggregate
behavior --- analogous to a per-expert logit lens applied across many documents. This is the
primary mode for proposing candidate expert specializations.

*Concept-restricted mode.* The dictionary is restricted to the token IDs corresponding to a
predefined concept word list (e.g., `numbers`, `countries`). SOMP then decomposes each
expert's activations against only those directions, and final EVR scores rank experts by how
well their output variation lies in that concept dictionary. This allows targeted queries such
as: _which experts' outputs are best represented by numeric token directions?_ It does not
measure routing frequency or causal responsibility. The concept word lists are defined in
`src/moe_interp/pursuit/concepts.py`.

The two modes are complementary: full-dictionary pursuit discovers specialists without prior
hypotheses, while concept-restricted pursuit quantifies specialization along a specific
semantic axis and is directly actionable for targeted interventions such as gate knockout
and down-weighting.

== From Specialization to Causation <sec:causal-methods>

Expert Pursuit is _correlational_: it reports which vocabulary directions an expert's output
aligns with, not whether that expert _causes_ a behavior. To test causation we build a small
circuit pipeline that, for a chosen concept, _localizes_ the experts the concept routes through
and then _intervenes_ during generation to remove it. The pipeline is concept-agnostic --- only
the target token set and prompts change --- and we run it on three concepts of decreasing
lexical sharpness: `countries`, `numbers`, and `offensive` (a harmful/sensitive-word proxy).
Throughout, the
correlational pursuit (SOMP) ranking from above is the association-only baseline. The reported
OLMoE causal experiments ran locally on Apple MPS.

The pipeline is a *selectors $times$ interventions* design: a selector proposes the concept's
experts, an intervention acts on them, and we ask both whether the _selector_ matters and whether
the effect is _necessity_ or merely _influence_.

The fused-experts kernel in OLMoE does not expose per-expert hidden neurons during a forward
pass, so the only differentiable, interventionable per-expert node is the *router gate*: tapping
`layer.mlp.experts.inputs[0]` yields the boundary tuple $(bold(h), "idx", bold(g))$ of hidden
states, selected expert indices, and gate weights. The gate-level operations below all act on
$bold(g)$; the direction-level controls act on the residual $bold(h)$.

=== Concept Probe and Prompts

We score a concept with a *concept-logit* probe: for a logit vector $bold(z)$ at the prediction
position, $ s_(cal(C))(bold(z)) = 1/(|cal(C)|) sum_(t in cal(C)) z_t - 1/v sum_(t=1)^v z_t $ <eq:conceptprobe>
where $cal(C)$ is the set of single-token concept-word variants (e.g. the `offensive` list)
and $v$ is the vocabulary size --- the mean concept-token logit relative to the row mean. We
complement this sensitive probe with a literal *word-fraction*: the share of generated
continuations that contain a concept word. For prompts we draw a split from RealToxicityPrompts
@gehman2020realtoxicityprompts, partitioning by each prompt's own toxicity score into a
high-toxicity _eliciting_ set and a matched low-toxicity _neutral_ set; the neutral set doubles
as a *collateral check* on the intervention (how far it drags down prompts that never elicited the
concept).

The `offensive` experiment uses RealToxicityPrompts for the gate-AtP identification set and a
disjoint held-out test set for intervention scoring. The stored `countries` and `numbers`
experiments instead use separately authored concept-eliciting identification and evaluation
prompts; their smaller evaluation sets make those comparisons exploratory rather than a matched
cross-concept benchmark. SOMP is fitted independently from stored RTP activations.

=== Selectors: Which Experts <sec:selectors>

We compare three ways to pick a concept's top-$k$ experts:

- *SOMP* (correlational) --- experts whose pursuit atoms most overlap the concept lexicon; the
  no-forward-pass, association-only baseline.
- *Gate-AtP* (causal approximation) --- one backward pass per prompt batch. Following the
  gradient-times-activation
  idea used in attribution patching @kramar2024atp, it estimates every
  expert's contribution to the metric from a first-order expansion: zeroing a gate ($g_e -> 0$)
  changes the metric by $approx - g_e dot (dif cal(L))/(dif g_e)$, so the expert's _contribution_
  --- how far the probe would drop on ablation --- is the negative of that,
  $ "AtP"(l,e) approx sum_("pos") g_e dot (dif cal(L)) / (dif g_e), quad cal(L) = sum_("prompt") s_(cal(C)), $ <eq:atp>
  where $g_e$ is the gate weight wherever expert $e$ fired. Sign: positive = the expert raises the
  concept score, so ablating it would lower it (the same sign as the gate-ablation grid @eq:patch,
  with which it correlates $r approx +0.69$; see @sec:results). This is our direct-effect selector,
  driving
  every intervention below; @app:atp-alg gives the batched procedure.
- *Random* (control) --- one deterministic draw of $k$ experts in the same layers as the AtP set.
  This controls layer footprint but does not estimate variability across random sets.

Gate-AtP is a first-order approximation of *exhaustive gate ablation* --- zeroing each
expert's gate in a separate forward pass and recording the probe change,
$ "PE"(l,e) = bb(E)_("prompt") [ s_(cal(C))(bold(z)_("base")) - s_(cal(C))(bold(z)_(- (l,e))) ], $ <eq:patch>
which is the exact effect of this particular gate-zeroing intervention but costs one forward pass
per routed layer--expert slot. We ran the ablation grid *once* to validate gate-AtP and the two
agreed closely (Pearson
$r approx 0.69$ pooled, $approx 0.93$ in the late layers; @sec:results), so the expensive sweep is
not part of the pipeline --- AtP provides a useful, imperfect ranking approximation at a fraction
of the cost. The
AtP grid spans all 16 layers $times$ 64 experts; a positive entry promotes the concept, a negative
entry _suppresses_ it.

=== Interventions: What We Do <sec:interventions>

Every intervention is *expert-level* --- it acts on the selected experts' post-top-$k$ routing
weight, never directly on the residual stream. It does not renormalize the remaining weights or
select replacement experts. Each
selected set is hit with one of two gate interventions, applied at every decoded step of greedy
generation and read out on the held-out prompts:

- *Knockout* (necessity) --- zero the gates of the top-$k$ experts. The simplest, scale-free test:
  is any sparse expert set _necessary_ for the concept?
- *Down-weighting* (dose--response) --- multiply the same gates by $s in (0,1)$ instead of zeroing
  them. Since the expert's contribution to the residual is $g_(t,e) dot bold(f)_e (bold(h)_t)$,
  scaling the live gate by $s$ scales that contribution at exactly the tokens routed to $e$ and
  leaves every other expert untouched. Sweeping $s in {0.9, 0.5, 0.25, 0}$ (a 10% down-weight
  through to full knockout at $s = 0$) turns the binary necessity test into a propensity-vs-strength
  curve, with per-prompt bootstrap error bars. Both interventions are implemented as a single gate
  scaling (`gate_scale_intervention`), knockout being the $s = 0$ endpoint.

Scoring is multi-signal: the mean probe value over each method's generated continuation (lower =
less concept-lexicon elevation), the literal word-fraction, the *neutral* prompts as a collateral
check, and a
*distinct-1* coherence guard (the ratio of unique unigrams; a healthy continuation sits around
$0.6$--$0.9$). This guard detects severe repetition but is not a general fluency metric. Because
each intervention is evaluated on the continuation it generates, the rollout score combines a
direct logit change with the effect of entering different subsequent contexts. The probe is
lexical: in particular, the `offensive` score is not an independent toxicity classifier.
