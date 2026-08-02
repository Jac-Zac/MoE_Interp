= Results <sec:results>

We ran Expert Pursuit on 10,000 TriviaQA questions using OLMoE-1B-7B-Instruct, capturing
last-token gated outputs for all 16 layers $times$ 64 experts. Of the 1,024 experts, only 334 had
sufficient activations (at least 5 routed documents) to analyze --- not because the rest are dead,
but because every TriviaQA prompt ends in the _same_ final token (the chat-template generation
suffix after "Answer:"), so the last-token readout always routes that position to one stable expert
support. The pile-10k extraction (completion-style, with lexically diverse final tokens) exercises
all 1,024 experts; we therefore report the per-expert EVR distribution and the logit-lens comparison
(@tab:lens, @sec:lens) on pile-10k, and read the discovery token-summaries in @tab:experts off the
TriviaQA run.

== Expert Specialization

Final EVR values (after $T = 25$ SOMP iterations) range from 0.041 to 0.285, with a median
of 0.076 across the 334 sufficiently-sampled experts (and 0.042 on the all-expert pile-10k
extraction used for @tab:lens and the cross-model check; see @tab:lens for the mean EVR at fewer
atoms, which is smaller). A low median EVR shows that the expert's output variance is poorly
captured by a small subspace of vocabulary directions. It is not, on its own, evidence of
_polysemanticity_ or superposition: the readout could simply be poorly aligned with the
unembedding dictionary. We therefore add a dictionary-geometry diagnostic in @sec:polysemy,
asking whether an expert's top atoms span many weakly aligned token families. Both observations
match the MoE interpretability literature, which reports that experts pack features in
superposition @lecomte2025sparsity and that single experts under-determine model behavior
@monosemanticpaths2026 @illusionspecialization2026. Nevertheless, a substantial minority of
experts exhibit coherent candidate specializations, mostly in the later layers. @tab:experts shows
representative examples across several categories identified by full-dictionary pursuit.

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
    [L15 E03], [Numbers / dates],  [35, 66, 2004, 150, 17, four, twenty, 123, 91],     [0.256],
    [L14 E37], [Numbers / dates],  [1997, 16, 1970, 26, June, 11, 19, 1930],           [0.104],
    [L13 E46], [Numbers],          [26, 63, 600, 105, 40, 10, 14, 160, 90],            [0.062],
    [L15 E49], [Geography],        [Japanese, American, Europe, South, Ukrainian, British, North], [0.127],
    [L11 E59], [Geography],        [Belgium, Britain, Maryland, Cleveland, Lithuanian], [0.086],
    [L12 E40], [Geography],        [Mediterranean, Welsh, Madrid, Africa, mountain, Castle], [0.044],
    [L15 E16], [Names],            [Ryan, Richard, Robert, Daniel, Garcia, James, John], [0.073],
    [L15 E38], [Biology],          [human, chemical, plant, metabolism, food, Animal, blood], [0.076],
    [L15 E59], [Entertainment],    [comic, music, debut, screen, thriller, play, hit], [0.071],
    [L14 E08], [Food],             [banana, jar, fruit, drinks, chicken, Apple],       [0.064],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Selected experts identified by full-dictionary pursuit on 10,000 TriviaQA documents. EVR is
the cumulative explained variance ratio after 25 SOMP iterations. Top tokens are the
highest-ranked readable atoms (sub-word fragments omitted).
  ],
) <tab:experts>

The selected coherent examples cluster in later layers (L12--L15). Earlier layers exhibit lower
EVR and less readable token lists under the *final* unembedding, but this does not by itself show
that they encode only lower-level features: the final unembedding is a less calibrated readout of
early residual states.

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
    [L15 E03], [10, 6, 100, 9, 2],                  [0.041],
    [L01 E09], [nineteen, fifty, three, one, thirty], [0.033],
    [L04 E14], [ninety, fifty, one, three, hundred],  [0.025],
    [L14 E37], [7, 3, five, 10, 4],                  [0.023],
    [L13 E55], [10, 2, 7, fourth, 0],                [0.022],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Top five experts by EVR\@10 under `numbers`-restricted pursuit on the stored RTP extraction.
  ],
) <tab:numbers>

Despite using a different dataset, the ranking recovers the same number
specialists as the full-dictionary TriviaQA run: L15 E03 ranks first and L14 E37 is again among the top
experts, and their unrestricted token lists consist almost entirely of numerals and year
tokens (@tab:experts). Concept-restricted pursuit therefore provides convergent, still
correlational evidence for these candidate specialists.

== Logit Lens vs SOMP: a Single Readout Under-Reads an Expert <sec:lens>

Our central methodological claim is that one direction per expert --- the standard *logit-lens*
readout @nostalgebraist2020logitlens --- is too coarse for a polysemantic expert, and that the
multi-atom SOMP basis recovers structure a single ranking misses. We test this directly. The
logit lens reads a *single* direction: it ranks tokens by the expert's *mean* activation,
$"top-"k(bold(D) macron(bold(e)))$, and its top token names one unembedding row whose EVR we
measure. SOMP instead selects a *basis* of atoms explaining the *variance* of the centered
activations. Both EVRs use the identical estimator (squared projection over total variance)
used inside the SOMP run, so they are directly comparable.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Readout*], [*EVR\@1*], [*EVR\@3*], [*EVR\@10*],
      table.hline(stroke: 0.5pt),
    ),
    [Logit lens (1 direction)], [0.0010], [---], [---],
    [SOMP (variance basis)],    [0.0020], [0.0056], [0.0146],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
EVR averaged over all 1,024 experts, computed on a pile (10k-document) extraction. The logit
lens reads one direction, so it has a single EVR (0.0010); SOMP's basis is shown at depths 1, 3,
and 10. Even atom-for-atom SOMP captures $approx 2 times$ the lens, and its 10-atom basis
$approx 14 times$, yet both stay under 2%.
  ],
) <tab:lens>

@tab:lens shows two things. First, the absolute EVR is small for both methods --- even ten
atoms explain under 2% of an expert's activation variance in the normalized unembedding
dictionary. This establishes weak low-dimensional vocabulary alignment, not polysemanticity by
itself. Second, SOMP captures $approx 2.1 times$ the
variance of the logit lens.
A single mean-direction ranking therefore under-reads variation by construction: it selects from
the expert's mean output rather than the directions along which centered outputs vary.
This is the per-expert analogue of the cross-layer finding that semantics in MoEs live in
distributed structure rather than any single component @monosemanticpaths2026, and it is the
empirical justification for preferring a sparse multi-atom basis over a one-shot logit lens.

== Atom-Family Coherence <sec:polysemy>

Polysemanticity is the semantic claim that one expert mixes unrelated concepts. We probe one
necessary signature in the
model's own readout geometry: every SOMP atom is a row of the unembedding, so two atoms are
"related" when their unembedding rows are aligned (cosine $gt.eq 0.4$) and unrelated when
near-orthogonal. For each expert we cluster its top-30 atoms (single-linkage at that threshold)
and measure the _largest-family share_ --- the fraction of atoms falling in the biggest cluster.
A coherent readout is dominated by one family (share near 1); a heterogeneous atom list splits
into many singletons (share near $1\/30$).

The median expert has a largest-family share of just *0.03*: its biggest aligned token group is
about one atom out of thirty. A genuine single-topic lexicon, by contrast --- the `numbers`
concept words, clustered identically --- sits at *0.37*, an order of magnitude tighter, and only
*4%* of experts have a coherent core of four or more related atoms. The effect is not an artifact
of SOMP's atom decorrelation: a truly single-topic expert yields a clustered atom set (the sharpest
expert, British/formal spelling at EVR 0.45, does retain an `amongst`/`among`/`whilst`/`realised`
core), whereas the typical expert does not. This thresholded, single-linkage diagnostic is
sensitive to the cosine threshold and lacks a frequency-matched null, so we treat it as evidence
of heterogeneous atom lists rather than a definitive polysemanticity measurement. The single
_highest_-EVR expert
(`L00 E06`, which fires on almost every prompt) is a complete grab-bag (`this`, `The`, `Kindle`,
`Sutton`, `History`, ...) --- high EVR is not monosemanticity. The readable specialization is often
a thin on-theme subset of a heterogeneous atom list. The
diagnostic is reproducible via
`scripts/atom_polysemanticity.py`.

== From Association to Causation: A Localizability Gradient <sec:causal>

The pursuit results above are correlational. We now test, on OLMoE, _which_ experts causally
drive a concept and whether acting on them removes it (@sec:causal-methods), running the same
selectors~$times$~interventions pipeline on three lexicons of decreasing sharpness:
`countries`, `numbers`, and harmful/sensitive content (`offensive`).

#emph[Data and protocol.] The `offensive` experiment uses RealToxicityPrompts
@gehman2020realtoxicityprompts, split by prompt toxicity into a gate-AtP identification set
($n_"train" = 100$) and a disjoint intervention set ($n_"test" = 64$). The stored `countries` and
`numbers` grids were identified on 60 separately authored eliciting prompts and evaluated on 18
eliciting plus 18 neutral prompts per concept. These lexical prompt sets are not matched to RTP,
and their complete source lists are absent from the current repository; those comparisons are
therefore exploratory and not fully reproducible from the current checkout. The SOMP selector is
the concept-restricted Expert-Pursuit ranking
(dictionary restricted to the concept lexicon, ranked by EVR\@10); the gate-AtP selector is the
grid of @eq:atp computed on the separate identification prompts. Intervention numbers are computed
on prompts not used for gate-AtP identification.

The point estimates suggest a _localizability gradient_ --- countries is sharper than numbers,
while the offensive lexicon has the weakest sparse handle. Within each prompt set, gate-AtP has
the largest knockout point estimate. Because the lexical experiments are small and use one fixed
layer-matched random draw, we do not interpret the cross-concept ordering as a calibrated ranking
or the selector differences as formal between-method significance tests.

=== Causal Localization Is a Per-Concept Gradient

Gate-AtP (@eq:atp) scores every $(l,e)$ with one backward pass per prompt batch, giving one signed
$16 times 64$ map per concept. The three maps differ qualitatively (@fig:atp-grids): for
`countries` the attribution concentrates in a handful of strong late-layer experts --- a sharp,
$approx 1%$-of-experts handle; for `numbers` the signal exists but is spread thinly across many
experts (distributed, leaky); for the offensive lexicon it is diffuse and low-magnitude
everywhere, with no sparse lever. This qualitative gradient organizes the section; the
interventions below test whether the point estimates follow the same ordering.

#figure(
  image("../figures/grid_atp_concepts.png", width: 100%),
  caption: [
Gate-AtP attribution maps ($16$ layers $times$ $64$ experts) for `countries`, `numbers`, and
`offensive` (harmful/sensitive-word proxy). Colour is the signed first-order gate-ablation effect
(@eq:atp); red
promotes the concept, blue suppresses it. The maps grow visibly more diffuse left-to-right:
`countries` concentrates on a few late-layer experts, `numbers` spreads thinly, and `offensive` has
no sparse lever --- the localizability gradient that the interventions trace.
  ],
) <fig:atp-grids>

For the offensive-lexicon probe specifically, the exhaustive gate-ablation grid (@eq:patch)
confirms the diffuse picture and
adds a twist invisible to a vocabulary readout: the largest direct gate effects are *not* confined
to the late layers where the pursuit specialists live. @tab:patch lists the nine with the largest
absolute effect --- they span layer~1 to layer~15, roughly half are *suppressors* (ablating them
_raises_ the offensive-word probe), the largest single effect is only $approx 0.04$ probe units,
and no expert
dominates. Pursuit's high-EVR examples cluster late in part because the final unembedding is an
uncalibrated readout of early-layer activations; the causal grid has no such mismatch. The tiny,
sign-mixed, depth-spread effects motivate the broader-budget ablations below.

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
    [L04 E14], [$-0.041$], [L15 E02], [$-0.032$], [L02 E30], [$+0.026$],
    [L15 E54], [$-0.024$], [L15 E56], [$-0.018$], [L14 E55], [$+0.018$],
    [L15 E47], [$-0.018$], [L01 E03], [$+0.017$], [L01 E49], [$+0.017$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Experts with the largest absolute gate-ablation effect (@eq:patch) on the offensive-word probe.
Positive = the expert promotes the probe (ablation lowers it); negative = suppressor. The effects
span early and late layers, unlike the late-layer pursuit specialists, and are uniformly small.
  ],
) <tab:patch>

=== Faithfulness: gate-AtP Approximates Exhaustive Gate Ablation

The gate-ablation grid is the exact reference for the same gate-zeroing intervention but costs one
forward pass per routed layer--expert slot. Gate-AtP --- one backward pass per prompt batch ---
tracks it moderately overall (pooled $r approx 0.69$) and *highly* in the late layers
($r approx 0.91$--$0.96$ for L12--L15), degrading only in the early layers ($r approx
0.30$--$0.49$) where per-expert effects are near zero and the first-order gradient is noisiest.
The late-layer agreement supports using the cheap AtP grid to rank candidates, while the weaker
early-layer agreement remains a limitation.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, left, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Selector*], [*Cost*], [*$r$ (pooled)*], [*$r$ (L12--15)*],
      table.hline(stroke: 0.5pt),
    ),
    [gate-AtP (gradient)],   [1 backward / batch],      [$+0.69$], [$+0.93$],
    [gate ablation (exact)], [1 forward / slot],        [---],     [---],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
One-off faithfulness check of gate-AtP against exact gate ablation (Pearson $r$ over scored
experts). The gradient proxy is moderately faithful pooled and highly faithful in late layers.
Correlation validates this first-order ranking approximation,
not gate-AtP as a general causal ground truth.
  ],
) <tab:faith>

#figure(
  image("../figures/patching_faithfulness.png", width: 100%),
  caption: [
Faithfulness of gate-AtP against exact gate ablation on the offensive-word probe. _Left:_ the
exhaustive ablation effect map. _Centre:_ gate-AtP vs ablation per slot (each point an $(l,e)$); the cloud
tightens toward the diagonal in the late layers. _Right:_ per-layer Pearson $r$, rising from
$approx 0.30$ early to $approx 0.96$ in the last layer --- AtP is most faithful exactly where the
controllable signal lives. The archived panel uses "patching" for this same gate-zeroing ablation.
  ],
) <fig:faithfulness>

=== Acting on the Causal Experts: Knockout and Down-Weighting

With the cheap selector validated we ask the decisive question: does acting on a concept's experts
remove it? Both interventions are *expert-level and gate-only* --- during greedy generation we scale
the router gate of the top-$k$ experts, either zeroing it (*knockout*, the necessity test) or
multiplying it by $s in (0,1)$ (*down-weighting*, the dose--response) --- and we score the held-out
concept-logit propensity (@eq:conceptprobe) against the un-intervened baseline (positive $Delta$ =
less of the concept). We report the top-$1%$ budget ($k = 10$ of 1,024 experts) and compare the
three selectors on all three concepts.

@tab:selector reports the central point estimates. Knocking out the gate-AtP experts lowers the
concept probe more than the SOMP or random sets on every prompt set, and the size of that drop
tracks how concentrated the concept's AtP map appears --- it reduces
$approx 21%$ of the country signal, $approx 7%$ of numbers, but only $approx 5%$ of the
offensive-word probe.
The paired baseline CIs exclude zero for gate-AtP on all three sets, but the report does not contain
paired gate-AtP-vs-random tests; accordingly, larger point estimates are not described as formal
selector wins. The word-level view also shows that even zeroing 103 layer--expert slots does not
make country words vanish. The intervention zeros already-selected weights without
renormalizing or selecting replacements. Robustness can arise because most affected tokens retain
other active experts and because later routing can change after the hidden state is perturbed; the
experiment does not directly isolate either mechanism.

#figure(
  table(
    columns: (1.6fr, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Concept* (baseline propensity)], [*gate-AtP*], [*SOMP*], [*random*],
      table.hline(stroke: 0.5pt),
    ),
    [`countries` ($2.38$)], [$bold(+0.49)$], [$+0.40$], [$+0.17$],
    [`numbers` ($3.70$)],   [$bold(+0.26)$], [$+0.05$], [$+0.03$],
    [`offensive` ($2.09$)], [$bold(+0.10)$], [$+0.03$], [$+0.06$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Top-$1%$ knockout ($k = 10$, gate zeroed) reduction in the concept-logit propensity per selector,
evaluation prompts; positive = lower rollout propensity. Gate-AtP has the largest point estimate
on each set. Its paired baseline 95% CIs are countries $[+0.29,+0.70]$, numbers
$[+0.10,+0.42]$, and offensive $[+0.00,+0.19]$ (rounded). These intervals test each method
against its own baseline, not selectors against each other. Shares of baseline are approximately
$21%$, $7%$, and $5%$, but the differently constructed prompt sets limit cross-concept comparison.
  ],
) <tab:selector>

Down-weighting instead of zeroing the gate gives a dose--response check. Sweeping the gate
multiplier $s in {0.9, 0.5, 0.25, 0}$ (a 10% down-weight through to full removal), the point
estimates generally grow with intervention strength, although several lexical-set intervals are
wide at $n = 18$. The full per-strength curves, with paired per-prompt bootstrap CIs, are produced by the
down-weight sweep and stored under `circuit/downweight/`.

@fig:generation gives one qualitative example per prompt set. These cherry-picked rollouts make
the intervention concrete but are not additional statistical evidence: the aggregate paired
scores and intervals in @tab:selector carry that role. They also show that a lower lexical probe
can accompany a factual error, repetition, or a changed topic, none of which distinct-1 reliably
detects.

#figure(
  text(size: 8.5pt, table(
    columns: (18%, 11%, 71%),
    inset: 2pt,
    align: (left, left, left),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header([*Prompt*], [*Set*], [*Greedy continuation excerpt*],
      table.hline(stroke: 0.5pt)),
    table.cell(rowspan: 4)[`countries`], [base], ["the village of Kherson, *Ukraine*"],
    [AtP], ["the land of *Arden* ..." #text(fill: rgb("#0a0"))[(invented)]],
    [SOMP], ["the village of *K village* ..."],
    [random], ["the village of Khersonos, *Ukraine*"],
    table.hline(stroke: 0.3pt),
    table.cell(rowspan: 4)[`numbers`], [base], ["*two* times *two* is *four* ..."],
    [AtP], ["*Twelve* times 12 is 12 squared" #text(fill: rgb("#0a0"))[(non-answer)]],
    [SOMP], ["eight times is eight ..."],
    [random], ["*four* times *four* is *sixteen* ..."],
    table.hline(stroke: 0.3pt),
    table.cell(rowspan: 4)[`offensive`], [base], ["... f\*\*\*ing islamophobes ..."],
    [AtP], ["just trying to survive in a world that hates us ..."],
    [SOMP], ["... f\*\*\*ing disgusting and offensive ..."],
    [random], ["You are a f\*\*\*ing piece of shit" #text(fill: rgb("#0a0"))[(repeated)]],
    table.hline(stroke: 0.8pt),
  )),
  caption: [
Selected generation excerpts under top-$1%$ gate knockout. These examples illustrate possible
failure modes and are not representative estimates; toxic text is censored.
  ],
) <fig:generation>

=== Association Does Not Guarantee Causal Responsibility

The correlational SOMP selector is the sharpest test of whether token-association implies causal
responsibility. On the sharply-localized
`countries` the SOMP set nearly matches gate-AtP under knockout (@tab:selector, $+0.40$ vs $+0.49$).
For `numbers` and the offensive lexicon, SOMP knockout falls back to the
random point estimate ($+0.05$ and $+0.03$, versus random's $+0.03$ and $+0.06$), while gate-AtP
has larger reductions. An expert's SOMP atoms tell us how its outputs vary in vocabulary space;
they do not establish that removing it changes the concept. The small samples and single random
draw preclude a stronger selector-comparison claim.

=== The Offensive-Lexicon Tail: No Strong Sparse Lever

The offensive lexicon sits at the bottom of the observed gradient. Knockout of the top-$1%$
gate-AtP experts lowers its rollout propensity
by only $+0.10$ ($approx 5%$ of the $2.09$ baseline; @tab:selector) --- the causal set still keeps
the largest point estimate, but its rounded 95% CI begins near zero and the experiment does not
directly compare selectors. The probe is a hand-built harmful/sensitive-word lexicon rather than
an independent toxicity measure, so these results do not establish that toxicity as a semantic
behavior is fully distributed. They show only that this probe has no strong sparse gate handle at
the tested budgets.

Taken together the three prompt sets suggest one gradient --- `countries` (sharp, $approx 1%$ handle)
$>$ `numbers` (distributed, leaky) $>$ `offensive` (diffuse). The gate gradient is useful for
ranking candidate handles, and the knockout point estimates track
the qualitative concentration of the maps. @tab:gradient collects the observations while keeping
the prompt-set mismatch in view.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Concept*], [*AtP map*], [*AtP knockout $Delta$*], [*AtP vs random*], [*Observation*],
      table.hline(stroke: 0.5pt),
    ),
    [`countries`], [sharp, $approx 1%$ late], [$bold(+0.49)$ ($approx 21%$)], [$+0.49$ vs $+0.17$], [strongest],
    [`numbers`],   [distributed],            [$bold(+0.26)$ ($approx 7%$)],  [$+0.26$ vs $+0.03$], [weaker],
    [`offensive`], [diffuse],                [$bold(+0.10)$ ($approx 5%$)],  [$+0.10$ vs $+0.06$], [weakest],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
The localizability gradient at a glance. For each concept: the shape of its gate-AtP map, the
top-$1%$ gate-AtP knockout reduction in the concept-logit propensity (absolute $Delta$ and share of
baseline), and the point estimate for the layer-matched random control. Prompt sources and sample
sizes differ, and no paired selector-vs-selector test is reported, so this is descriptive evidence
rather than a calibrated cross-concept comparison.
  ],
) <tab:gradient>

=== Stress-Testing Robustness: Cumulative and Co-Firing Ablations

Two further checks test robustness at larger budgets. A *cumulative ablation curve* knocks out the
top-$j$ experts of each selector for growing $j$ and scores the held-out offensive-word probe
(@tab:sufficiency; positive means a lower probe). The gate-AtP set lowers the probe _monotonically_
at the reported budgets
but the reduction only reaches $+0.21$ at $j = 103$ (10% of all experts), and it is the _only_
selector whose effect is statistically resolved (95% bootstrap CI $[+0.11, +0.30]$ at $j = 103$);
SOMP and random straddle zero at every budget, and SOMP raises the probe at large $j$.
Distinct-1 holds at $approx 0.83$ throughout, ruling out severe repetition but not other quality
changes. A
*co-firing group ablation* agrees: padding the 15 AtP experts with their four top co-firing
neighbours each (69 experts) deepens the reduction to $+0.16$ (probe $2.07 -> 1.91$) versus only
$+0.02$ for a size-matched random group. These interventions show robustness of the probe to broad
gate ablation. They do not distinguish concurrent-expert redundancy from altered later routing or
other distributed computation.

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Selector*], [$j = 1$], [$j = 10$], [$j = 50$], [$j = 103$],
      table.hline(stroke: 0.5pt),
    ),
    [gate-AtP], [$+0.01$], [$+0.09$], [$+0.11$], [$bold(+0.21)$],
    [SOMP],     [$0.00$],  [$+0.03$], [$-0.05$], [$-0.10$],
    [random],   [$+0.03$], [$+0.03$], [$-0.02$], [$-0.04$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Cumulative ablation curve: reduction in the offensive-lexicon probe (baseline $+2.07$; positive =
lower probe) as
the top-$j$ experts of each selector are knocked out, held-out $n_"test" = 64$, distinct-1
$approx 0.83$ throughout. Only gate-AtP falls monotonically, and only its $j = 103$ point is
bootstrap-significant (95% CI $[+0.11, +0.30]$); SOMP and random straddle zero. Knocking out even
$10%$ of all layer--expert slots leaves most of the probe intact.
  ],
) <tab:sufficiency>

== Cross-Model Check: GPT-OSS-20B

We re-ran the pipeline on a second, larger model --- `openai/gpt-oss-20b`, a 24-layer sparse MoE
with 32 local experts per layer, top-4 routing, and hidden size 2880 @openai2025gptoss --- to test
whether the observations are architecture-specific. Three findings recur. (1) *Weak sparse
dictionary alignment*: the median per-expert EVR is $0.042$ over 748 sufficiently sampled
layer--expert slots (pile-10k extraction), essentially identical to OLMoE's $0.042$ over all
1,024 slots on the same dataset. A small vocabulary basis under-explains output variance in both
models; this is not by itself a polysemanticity measure.
Qualitatively, though, gpt-oss's experts are _less lexically interpretable_: even on TriviaQA its
highest-EVR experts are dominated by multilingual fragments, code, and punctuation rather than the
clean semantic categories OLMoE shows (@tab:experts), consistent with its 200k multilingual vocabulary
and coarser top-4 routing --- so the clean per-expert _token_ summaries are more evident in OLMoE.
(2) *AtP faithfulness*: gate-AtP tracks exhaustive gate ablation more closely here than on OLMoE
(pooled Pearson $r approx 0.77$), supporting the first-order approximation across these two
architectures. (3) *Intervention robustness*: on a held-out offensive-probe split, top-8 gate-AtP
knockout lowers the probe by
$approx 0.4$ for AtP versus $approx 0$ for random, yet with coherence loss (distinct-1
$approx 0.65$) and a sizeable collateral drop on the _neutral_ prompts ($approx 0.2$), so the
effect is neither clean nor specific. With only $n_"train" = 50$ / $n_"test" = 30$ prompts the
per-selector intervention picture is noisy, so we read GPT-OSS as a replication of the dictionary-
alignment and gate-AtP-correlation observations rather
than a second intervention study.
