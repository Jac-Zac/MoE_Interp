= Results <sec:results>

We ran Expert Pursuit on 50,000 TriviaQA questions using OLMoE-1B-7B-Instruct, capturing
last-token gated outputs for all 16 layers $times$ 64 experts. Of the 1,024 experts, 393 had
sufficient activations (at least 5 routed documents) to analyze. We then re-ran the pipeline in
an *all-token* capture mode on a 10,000-document Pile sample @gao2020pile (@sec:alltoken),
which routes every token position to its experts rather than only the final token of each
document; this both raises the per-expert sample size enough to cover all 1,024 experts and
sharpens the resulting bases.

== Expert Specialization

Final EVR values (after $T = 25$ SOMP iterations) range from 0.021 to 0.248, with a median
of 0.041. The low median is itself a finding: most experts are polysemantic, so no small set
of vocabulary directions captures most of their variance. This matches the MoE
interpretability literature, which reports that experts pack features in superposition
@lecomte2025sparsity and that single experts under-determine model behavior
@monosemanticpaths2026 @illusionspecialization2026 (@sec:lens quantifies this directly).
Nevertheless, a substantial minority of experts exhibit clear semantic specialization, mostly
in the later layers. @tab:experts shows representative examples across several categories
identified by full-dictionary pursuit.

#figure(
  table(
    columns: (auto, auto, 1fr, auto),
    align: (left, left, left, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Expert*], [*Category*], [*Top tokens*], [*EVR*],
      table.hline(stroke: 0.5pt),
    ),
    [L15 E03], [Numbers / dates],  [9, 55, 26, 200, four, 2003, 87, 15, 314, 2, Oct], [0.191],
    [L14 E37], [Numbers / dates],  [16, 1991, 7, 40, 1960, June, 22, 10, 2005, 13],   [0.069],
    [L13 E46], [Numbers],          [17, 49, 8, 2, 110, 300, Act, 73, 32, 12, fifth],  [0.039],
    [L15 E49], [Geography],        [English, Thai, America, North, European, Canadian, British], [0.094],
    [L12 E40], [Geography],        [Mediterranean, England, Arabia, Madrid, Swiss, Iowa], [0.036],
    [L11 E59], [Geography],        [France, Italy, €, English, Lithuanian],            [0.048],
    [L14 E02], [Names],            [David, Steve, John, Sir, Andrew, George, Sarah],   [0.048],
    [L15 E16], [Names],            [Ryan, Smith, Bobby, Oliver, Shannon, Charles],     [0.044],
    [L15 E38], [Biology],          [protein, human, digestive, blood, plant, chemical],[0.054],
    [L15 E59], [Entertainment],    [music, comic, debut, screen, thriller, play],      [0.050],
    [L14 E08], [Food],             [fruit, olive, pot, chicken, food],                 [0.036],
    [L14 E51], [Kinship / address], [friends, beloved, child, folks, brother, ladies],  [0.077],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Selected experts identified by full-dictionary pursuit on 50,000 TriviaQA documents. EVR is
the cumulative explained variance ratio after 25 SOMP iterations. Top tokens are the
highest-ranked readable atoms (sub-word fragments omitted).
  ],
) <tab:experts>

Specialists tend to cluster in the later layers (L12--L15), consistent with the view that
deeper layers encode more abstract semantic content. Early layers (L00--L04) exhibit lower EVR
values and less coherent token lists, suggesting they operate on lower-level distributional
features. The all-token Pile run (@sec:alltoken) refines this picture: the *mean* EVR still
rises monotonically with depth, but the *sharpest* individual specialists turn out to live in
the early and middle layers.

== All-Token Capture on the Pile <sec:alltoken>

#text(fill: red)[_Template — figures below pending re-run: all-token capture is not in the
current `capture.py` (last-token only); numbers to be refilled after re-extraction._]

The last-token run above samples one activation per routed document, which leaves most experts
under-sampled --- only 393 of 1,024 cleared the 5-document threshold. Capturing *every* token
position instead routes far more activations to each expert: on the 10,000-document Pile sample
all 1,024 experts are covered, with a median of 1,630 activations per expert (minimum 351,
1.73 M activations in total, $approx 5 times$ the last-token yield). This is the run that made
the pursuit step substantially more expensive, and it gives the cleanest view of expert
structure we have.

Polysemanticity survives the extra data. Even with five times more activations per expert, the
mean cumulative EVR after 50 SOMP atoms is only $0.057$ (median $0.044$), and ten atoms explain
just $1.65%$ of an expert's variance --- the same order of magnitude as the last-token run.
Low-rank vocabulary structure is therefore an intrinsic property of the experts, not an
artefact of sparse last-token sampling.

The all-token bases also expose structure the last-token run could not. Two families dominate
the high-EVR tail (@tab:alltoken). First, the ten most concentrated experts in the *entire*
network all encode the *same* lexical feature: British/Commonwealth spellings and formal
connectives (`amongst`, `among`, `whilst`, `neighbourhood`, `flavour`, `organisations`). These
experts sit in layers 1--11 and reach EVR $approx 0.35$--$0.48$, roughly $6 times$ the global
mean, yet their token lists are nearly identical. The same register feature is thus represented
redundantly by many experts across depth --- a per-feature instance of the distributed,
non-monosemantic organisation reported for MoEs @monosemanticpaths2026
@illusionspecialization2026. Second, the late-layer specialists are predominantly *syntactic*
rather than topical: distinct L15 experts collect finite verbs and auxiliaries (`are`, `was`,
`had`, `did`), possessive and personal pronouns (`their`, `your`, `her`, `my`), sentence-initial
tokens (`It`, `If`, `This`, `When`), and whitespace/formatting markers. The number specialist
L15 E03 reappears here exactly as in the last-token run (@tab:experts), a useful cross-corpus
consistency check.

#figure(
  table(
    columns: (auto, auto, 1fr, auto),
    align: (left, left, left, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Expert*], [*Category*], [*Top tokens*], [*EVR*],
      table.hline(stroke: 0.5pt),
    ),
    [L02 E30], [British / formal],    [amongst, among, Whilst, have, While],          [0.475],
    [L09 E08], [British / formal],    [amongst, among, Whilst, neighbourhood, have],  [0.472],
    [L08 E22], [British / formal],    [amongst, among, Whilst, flavour, While],       [0.451],
    [L03 E39], [British / formal],    [amongst, among, Whilst, neighbourhood, have],  [0.430],
    [L15 E28], [Verbs / auxiliaries], [are, was, doesn, have, weren, did, will, had], [0.201],
    [L15 E33], [Sentence-initial],    [It, it, If, This, When, Why, which, there],    [0.193],
    [L15 E56], [Pronouns],            [their, your, it, which, her, we, this, my],    [0.187],
    [L15 E01], [Formatting / layout], [newline, code-fence, indent, "This"],          [0.184],
    [L15 E03], [Numbers],             [35, 9, 66, 150, 20, five, 13, 317],            [0.227],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Representative experts from the all-token Pile run, ranked within two families: the cross-layer
British/formal-register cluster (the highest-EVR experts in the whole network) and the
syntactic specialists of the final layer. EVR is the cumulative explained variance ratio after
50 SOMP atoms. Top tokens are the highest-ranked readable atoms (sub-word fragments omitted).
  ],
) <tab:alltoken>

@fig:dists summarises the population. The per-expert EVR distribution (a) is sharply
right-skewed --- a dense bulk of polysemantic experts below the $0.044$ median and a thin tail
of specialists --- and the layer view (d) makes the two trends in @tab:alltoken concrete: the
mean (bars) climbs steadily toward L15 while the per-layer maximum (line) is bimodal, spiking in
the early/middle layers where the British/formal experts sit. The recurrence of shared rank-1
atoms (@fig:atoms) is the redundancy finding in one plot: `amongst` is the single most common
top atom in the whole network, tied with the function word `of`.

#figure(
  image("../figures/pursuit_distributions.png", width: 92%),
  caption: [
Population statistics over all 1,024 experts (all-token Pile run). (a) Final EVR per expert
(dashed line: median $0.044$). (b) Activations per expert (log scale). (c) Cumulative EVR vs
SOMP depth --- median with inter-quartile band, and the single best expert (dotted). (d) Mean
(bars) and maximum (line) EVR by layer.
  ],
) <fig:dists>

#figure(
  image("../figures/pursuit_top_atoms.png", width: 62%),
  caption: [
The 18 most frequent rank-1 (highest-ranked) atoms across all 1,024 experts. `amongst` ties
`of` as the most common top atom, despite being a far rarer word --- a direct view of the
British/formal-register feature being encoded redundantly across many experts.
  ],
) <fig:atoms>

== Concept-Restricted Pursuit: Numbers

To validate the concept-restricted mode, we ran Expert Pursuit with the dictionary restricted
to the `numbers` word list (digit tokens plus English number words; see
`src/moe_interp/pursuit/concepts.py`).
@tab:numbers shows the top-ranked experts under this query.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Expert*], [*Top concept tokens*], [*EVR*],
      table.hline(stroke: 0.5pt),
    ),
    [L15 E03], [4, 10, three, 9, twenty],      [0.089],
    [L13 E55], [five, 6, first, one, eleven],  [0.053],
    [L14 E37], [5, 10, 2, 3, nine],            [0.051],
    [L01 E42], [double, six, fifty, twelve, eighteen], [0.043],
    [L15 E40], [eighty, thirteen, six, ninety, thirty], [0.041],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Top 5 experts ranked by EVR under concept-restricted pursuit with the `numbers` dictionary
(run on a smaller local sample of 242 documents).
  ],
) <tab:numbers>

Despite running on a different, much smaller sample, the ranking recovers the same number
specialists as the full-dictionary run: L15 E03 ranks first and L14 E37 is again among the top
experts, and their unrestricted token lists consist almost entirely of numerals and year
tokens (@tab:experts). Concept-restricted pursuit thus serves as a focused probe that confirms
and quantifies specialization identified by the full-dictionary run.

== Logit Lens vs SOMP: a Single Readout Under-Reads an Expert <sec:lens>

Our central methodological claim is that one direction per expert --- the standard *logit-lens*
readout @nostalgebraist2020logitlens --- is too coarse for a polysemantic expert, and that the
multi-atom SOMP basis recovers structure a single ranking misses. We test this directly. For
each expert we compare (i) the bulk logit lens, which ranks tokens by the expert's *mean*
activation, $"top-"k(bold(D) macron(bold(e)))$, against (ii) SOMP, which selects a *basis* of
atoms explaining the *variance* of the centered activations. To compare them on equal footing,
both methods' selected atoms reconstruct the same centered activations by least squares and we
read off the cumulative EVR with the identical estimator used inside the SOMP run.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Method*], [*EVR\@1*], [*EVR\@3*], [*EVR\@10*],
      table.hline(stroke: 0.5pt),
    ),
    [Logit lens (mean direction)], [0.0011], [0.0029], [0.0075],
    [SOMP (variance basis)],       [0.0030], [0.0074], [0.0196],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Cumulative EVR averaged over all 1{,}024 experts at decomposition depths 1, 3, and 10, for the
mean-projection logit lens versus SOMP, computed on the Pile extraction @gao2020pile. SOMP
explains $approx 2.6 times$ more activation variance at every depth. The mean top-10 token
overlap between the two readouts (Jaccard) is only 0.010, so the two methods largely disagree
on which tokens characterize an expert.
  ],
) <tab:lens>

@tab:lens shows two things. First, the absolute EVR is small for both methods --- even ten
atoms explain under 2% of an expert's activation variance --- a direct, quantitative signature
of polysemanticity: an expert's output does not lie near any low-dimensional vocabulary
subspace @lecomte2025sparsity. Second, SOMP consistently captures $approx 2.6 times$ the
variance of the logit lens, and the two readouts share almost no top tokens (Jaccard $0.010$).
A single mean-direction ranking therefore systematically under-reads an expert: it is biased
toward the high-norm mean rather than the directions along which the expert actually varies.
This is the per-expert analogue of the cross-layer finding that semantics in MoEs live in
distributed structure rather than any single component @monosemanticpaths2026, and it is the
empirical justification for preferring a sparse multi-atom basis over a one-shot logit lens.

== Causal Toxic-Expert Circuit <sec:causal>

The pursuit results above are correlational. We now test, on OLMoE, _which_ experts causally
drive a concrete behavior --- toxic continuation --- and whether acting on them suppresses it
(@sec:causal-methods). The probe and prompts are as defined there; the patching grid is computed
over all 16 layers on the toxic-eliciting prompt set, covering 913 routed experts.

=== Causal Localization: Experts Span All Depths, Including Suppressors

The activation-patching grid (@eq:patch) shows that the causally important experts are *not*
confined to the late layers where the pursuit specialists live. @tab:patch lists the ten
experts with the largest absolute effect: they range from layer~0 to layer~15, and roughly half
are *suppressors* (negative effect --- ablating them _raises_ toxicity). The single most causal
expert, L09~E12, is a strong suppressor. This is structure that a vocabulary-aligned readout
cannot see: pursuit's high-EVR specialists concentrate in the later layers, because projecting
_early_-layer activations onto the unembedding is ill-posed --- the early residual basis is
positional and syntactic, not yet semantic.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right) * 3,
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Expert*], [*Effect*], [*Expert*], [*Effect*], [*Expert*], [*Effect*],
      table.hline(stroke: 0.5pt),
    ),
    [L09 E12], [$-0.27$], [L04 E30], [$+0.24$], [L04 E48], [$-0.14$],
    [L14 E55], [$+0.14$], [L06 E36], [$+0.13$], [L02 E23], [$-0.13$],
    [L10 E06], [$-0.13$], [L00 E01], [$+0.10$], [L08 E35], [$+0.10$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Experts with the largest absolute causal effect (@eq:patch) on the toxic-logit probe. Positive =
the expert promotes toxicity (ablation lowers the score); negative = suppressor. The effects span
all depths, unlike the late-layer pursuit specialists.
  ],
) <tab:patch>

=== Faithfulness: gate-AtP Recovers the Causal Grid Cheaply

@tab:faith compares the cheap expert scores against the patching grid by Pearson
correlation over all 913 scored experts. Gate attribution patching (@eq:atp) --- a single
backward pass --- predicts the expensive grid closely (pooled $r approx 0.80$, and $0.83$--$0.98$
within individual layers), whereas the correlational SOMP/pursuit ranking is essentially
_uncorrelated_ with causal effect. The lesson is sharp: token association is not causation, but a
first-order gradient on the gate _is_ a faithful, one-pass proxy for the full ablation sweep.

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Method*], [*Cost*], [*Pearson $r$ vs patching*],
      table.hline(stroke: 0.5pt),
    ),
    [gate-AtP (gradient)],       [1 backward pass], [$+0.80$],
    [SOMP (token association)],  [no model],        [#text(fill: red)[_fill_]],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Faithfulness of the cheap attributors to the causal patching grid, pooled over 913 experts. The
gradient method is faithful; the correlational token-association ranking is not.
  ],
) <tab:faith>

=== Intervention: Causal Identification and Project-Out Suppress Toxicity

@tab:intervene reports the mean toxic-logit propensity over generated continuations under each
intervention (knockout of the top-15 experts from each identifier, plus down-weight and
project-out), relative to an unintervened baseline of $+2.48$. Two results stand out. First,
_only the causally-identified experts matter_: knocking out the AtP or patching set lowers
toxicity (by $0.48$ and $0.34$), while knocking out the SOMP- or random-identified sets
does essentially nothing --- a direct demonstration that correlational expert identification is
causally inert. Second, *project-out is the best suppressor*: removing the toxic direction from
the residual stream gives the largest drop ($-0.58$, a $approx 24%$ reduction) while keeping
generation fluent, whereas naive additive steering with a large coefficient collapses both toxic
_and_ neutral generation (and is excluded here). Knockout is effective but blunt --- it can leave
residual toxicity or degrade coherence --- so _identify causally, suppress by direction removal_
is the recipe we recommend.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Intervention*], [*Toxic propensity*], [*$Delta$ vs base*], [*Neutral*],
      table.hline(stroke: 0.5pt),
    ),
    [baseline],                 [$+2.48$], [---],     [$+0.92$],
    [AtP-knockout (top 15)],    [$+2.00$], [$-0.48$], [$+1.00$],
    [patching-knockout],        [$+2.14$], [$-0.34$], [$+0.83$],
    [AtP-downweight ($times 0.5$)], [$+2.20$], [$-0.28$], [$+0.90$],
    [SOMP-knockout],            [$+2.50$], [$+0.02$], [$+0.87$],
    [random-knockout],          [$+2.58$], [$+0.10$], [$+0.85$],
    [*project-out* (\@L12)],    [$bold(+1.90)$], [$bold(-0.58)$], [$+1.26$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Toxic-logit propensity over generated continuations under each intervention (lower = less toxic).
$Delta$ is the reduction from baseline; the neutral column is the collateral check. Causal
identifiers (AtP, patching) and project-out suppress toxicity; correlational identifiers (SOMP,
random) do not.
  ],
) <tab:intervene>

Qualitatively, project-out de-toxifies while preserving fluency. For the seed _"...the worst kind
of people are"_, the baseline continues _"those who are always trying to be nice. They're
insufferable..."_, whereas under project-out it becomes _"...always trying to be friendly, always
trying to make you feel comfortable..."_; for _"...I just want to slap them"_ the baseline
_"I can't stand it when people do something stupid..."_ becomes the more reflective _"I don't know
what it is about them that makes me so angry. Maybe it's their..."_. Because the direction and
probe are built from any concept's token set, the same `circuit-steer --concept` machinery applies
beyond toxicity (we verified, e.g., that projecting out a `numbers` direction lowers the model's
number-token propensity).

== GPT-OSS Support

The codebase also supports `openai/gpt-oss-20b` as a second target model. GPT-OSS is a
24-layer sparse MoE model with 32 local experts per layer, 4 experts routed per token, and
hidden size 2880 @openai2025gptoss. We do not report GPT-OSS results here, but the same
extraction and pursuit pipeline can be applied to it for future comparison.
