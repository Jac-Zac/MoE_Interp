= Methods <sec:methods>

We adapt Head Pursuit @basile2025headpursuit from attention heads to MoE experts. Where
Head Pursuit decomposes per-head residual stream contributions, we decompose per-expert gated
outputs using the same SOMP-based sparse coding framework.

== Dataset

We use TriviaQA @joshi2017triviaqa (RC configuration, train split), following the Head
Pursuit setup. Each question is one document. Questions are wrapped in the model's chat
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

For each expert, we run SOMP with the L2-normalized unembedding matrix as dictionary and
$T = 25$ iterations. This produces a ranked list of vocabulary tokens that best explain the
expert's variance across questions, along with cumulative EVR scores.

== Analysis Modes

The pipeline supports two complementary analysis modes, and either can run over a
word-augmented dictionary in which multi-token words are appended as averaged atoms on top of
the base vocabulary and re-normalized, so their direction --- not their smaller norm ---
determines their influence during SOMP.

*Full-dictionary mode.* SOMP searches the entire vocabulary ($v approx 50{,}000$ tokens). The
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
semantic axis and is directly actionable for targeted interventions such as activation
steering.

== From Specialization to Causation <sec:causal-methods>

Expert Pursuit is _correlational_: it reports which vocabulary directions an expert's output
aligns with, not whether that expert _causes_ a behavior. To test causation we build a small
circuit pipeline around a concrete target behavior --- toxic text generation --- in two stages:
we _localize_ the experts that are causally responsible, then _intervene_ during generation to
suppress the behavior, using the correlational pursuit (SOMP) ranking from above as the
association-only baseline to beat. The same machinery is concept-agnostic: only the target token
set and prompts are toxicity-specific. All experiments run locally on Apple MPS.

The fused-experts kernel in OLMoE does not expose per-expert hidden neurons during a forward
pass, so the only differentiable, interventionable per-expert node is the *router gate*: tapping
`layer.mlp.experts.inputs[0]` yields the boundary tuple $(bold(h), "idx", bold(g))$ of hidden
states, selected expert indices, and gate weights. Every causal operation below acts on $bold(g)$.

=== Toxicity Probe and Prompts

We score toxicity with a *toxic-logit* probe: for a logit vector $bold(z)$ at the prediction
position, $ s_("tox")(bold(z)) = 1/(|cal(T)|) sum_(t in cal(T)) z_t - 1/v sum_(t=1)^v z_t $ <eq:toxprobe>
where $cal(T)$ is a set of single-token offensive words (from the `offensive` concept list) and
$v$ is the vocabulary size --- the mean toxic-token logit relative to the row mean. For prompts
we draw a split from RealToxicityPrompts @gehman2020realtoxicityprompts, partitioning by each
prompt's own toxicity score into a high-toxicity _eliciting_ set and a matched low-toxicity
_neutral_ set. The eliciting set drives the patching, attribution, and intervention experiments;
a diff-of-means between the two sets isolates the toxic direction used for project-out.

=== Activation Patching (causal ground truth)

For every routed $(l,e)$ we run one forward pass with the expert's gate zeroed wherever it was
selected, and record the change in the probe at the last token, averaged over prompts:
$ "PE"(l,e) = bb(E)_("prompt") [ s_("tox")(bold(z)_("base")) - s_("tox")(bold(z)_(- (l,e))) ] $ <eq:patch>
A positive effect means the expert _promotes_ toxicity (ablating it lowers the score); a negative
effect marks a _suppressor_. This yields a $16 times 64$ causal grid, at a cost of one forward
pass per routed expert.

=== Gate Attribution Patching (gate-AtP)

Activation patching is exact but expensive. Attribution patching @kramar2024atp estimates the
entire grid from a single backward pass via a first-order expansion around the gate-to-zero
intervention:
$ "AtP"(l,e) approx - sum_("pos") g_e dot (dif cal(L)) / (dif g_e) $ <eq:atp>
where $cal(L) = sum_("prompt") s_("tox")$ and $g_e$ is the gate weight wherever expert $e$ fired.
We quantify how well this cheap estimate reproduces the patching ground truth by the Pearson
correlation of the two per-expert grids.

=== Interventions

To suppress toxicity at generation time we compare three families of intervention, applied at
every decoded step:
- *Knockout* --- zero the gates of the top-$k$ identified experts (we vary which identifier
  supplies the set: AtP, patching, SOMP, or a random control).
- *Down-weight* --- scale those gates by a factor (a softer knockout).
- *Project-out* --- remove the toxic direction's component from the residual stream at a layer,
  $bold(h) <- bold(h) - (bold(h) dot hat(bold(v))) hat(bold(v))$, leaving every
  orthogonal feature untouched. This is a non-destructive variant of activation steering
  @turner2023activation; $bold(v)$ is either the diff-of-means toxic direction or the unembedding
  concept direction $bold(d)_("tox")$.
Each method is scored by greedy generation under the intervention: the mean probe value over the
continuation (lower is less toxic) plus an offensive-word rate, with the neutral prompts as a
collateral check. Because the direction and probe can be built from any concept's token set, the
intervention generalizes to arbitrary concepts.
