= Methods <sec:methods>

We adapt Head Pursuit @basile2025headpursuit from attention heads to MoE experts. Where
Head Pursuit decomposes per-head residual stream contributions, we decompose per-expert gated
outputs using the same SOMP-based sparse coding framework.

== Dataset

We use TriviaQA @joshi2017triviaqa (RC configuration, train split, $n = 10,000$), following
the Head Pursuit setup. Each question is one document. Questions are wrapped in the model's chat
template without any additional QA prompt --- only the raw question text is presented to the
model.

== Model and Activation Extraction

We target OLMoE-1B-7B-Instruct @muennighoff2024olmoe: 16 layers, 64 experts per layer,
top-8 routing, $d = 2048$. Using `nnsight` for model tracing, we capture for each token: (1)
the router's top-$k$ expert indices and gating weights, and (2) the raw FFN output
$f_e (bold(x)_i)$ from each selected expert.

To keep the capture aligned with the model's positional encoding, prompts are traced in
right-padded batches and we extract the last real token from each prompt. The gated expert
output is then multiplied by the router weight and passed through an approximate final
RMSNorm using the residual-stream second moment, matching the scale of the model's final
representation.

== Aggregation

For each expert $e$ at layer $l$, we compute the gated output at the last token position for
each document $j$:

$ bold(e)_(e,l)^j = g_e (bold(x)_j) dot f_e (bold(x)_j) $ <eq:expert-agg>

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
primary mode for discovering what each expert specializes in.

*Concept-restricted mode.* The dictionary is restricted to the token IDs corresponding to a
predefined concept word list (e.g., `numbers`, `countries`). SOMP then decomposes each
expert's activations against only those directions, and final EVR scores rank experts by how
strongly they respond to that concept. This allows targeted queries such as: _which experts
are most active on numeric content?_ The concept word lists are defined in
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
lexical sharpness: `countries`, `numbers`, and `offensive` (toxicity). Throughout, the
correlational pursuit (SOMP) ranking from above is the association-only baseline to beat. All
experiments run locally on Apple MPS.

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
where $cal(C)$ is the set of single-token concept words (e.g. the `offensive` list for toxicity)
and $v$ is the vocabulary size --- the mean concept-token logit relative to the row mean. We
complement this sensitive probe with a literal *word-fraction*: the share of generated
continuations that contain a concept word. For prompts we draw a split from RealToxicityPrompts
@gehman2020realtoxicityprompts, partitioning by each prompt's own toxicity score into a
high-toxicity _eliciting_ set and a matched low-toxicity _neutral_ set; the neutral set doubles
as a *collateral check* on the intervention (how far it drags down prompts that never elicited the
concept).

Crucially, every selector (gate-AtP, SOMP ranking) is fit on a _train_ split of the prompts and
every intervention is then scored on a _disjoint held-out test_ split. This avoids the identify-and-score-on-the-same-prompts circularity that would otherwise
inflate any causally-selected method against the correlational baseline.

=== Selectors: Which Experts <sec:selectors>

We compare three ways to pick a concept's top-$k$ experts:

- *SOMP* (correlational) --- experts whose pursuit atoms most overlap the concept lexicon; the
  no-forward-pass, association-only baseline.
- *Gate-AtP* (causal) --- one backward pass. Attribution patching @kramar2024atp estimates every
  expert's contribution to the metric from a first-order expansion: zeroing a gate ($g_e -> 0$)
  changes the metric by $approx - g_e dot (dif cal(L))/(dif g_e)$, so the expert's _contribution_
  --- how far the probe would drop on ablation --- is the negative of that,
  $ "AtP"(l,e) approx sum_("pos") g_e dot (dif cal(L)) / (dif g_e), quad cal(L) = sum_("prompt") s_(cal(C)), $ <eq:atp>
  where $g_e$ is the gate weight wherever expert $e$ fired. Sign: positive = the expert raises the
  concept score, so ablating it would lower it (the same sign as the patching grid @eq:patch, with
  which it correlates $r approx +0.69$; see @sec:results). This is our causal selector, driving
  every intervention below; @app:atp-alg gives the one-pass procedure.
- *Random* (control) --- $k$ random experts in the same layers as the AtP set, isolating whether
  it has to be _these_ experts.

Gate-AtP is a first-order approximation of *exhaustive activation patching* --- zeroing each
expert's gate in a separate forward pass and recording the probe change,
$ "PE"(l,e) = bb(E)_("prompt") [ s_(cal(C))(bold(z)_("base")) - s_(cal(C))(bold(z)_(- (l,e))) ], $ <eq:patch>
which is the exact causal effect but costs one forward pass per routed expert ($approx 64 times$
more). We ran the patching grid *once* to validate gate-AtP and the two agreed closely (Pearson
$r approx 0.69$ pooled, $approx 0.93$ in the late layers; @sec:results), so the expensive sweep is
not part of the pipeline --- AtP gives effectively the same ranking at a fraction of the cost. The
AtP grid spans all 16 layers $times$ 64 experts; a positive entry promotes the concept, a negative
entry _suppresses_ it.

=== Interventions: What We Do <sec:interventions>

Every intervention is *expert-level* --- it acts on the selected experts' router gate, never on the
residual stream --- so a positive effect is attributable to those experts and nothing else. Each
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

Scoring is held-out and multi-signal: the mean probe value over the continuation (lower = less
concept), the literal word-fraction, the *neutral* prompts as a collateral check, and a
*distinct-1* coherence guard (the ratio of unique unigrams; a healthy continuation sits around
$0.6$--$0.9$). The coherence guard is what separates genuine concept removal from a method that
merely *degrades the text into garbage* --- a probe drop with a collapsed distinct-1 is not a
clean intervention. Because the probe is built from any concept's token set, the whole pipeline
generalizes across concepts.
