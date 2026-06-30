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
atoms, which is smaller). A low median EVR is a _superposition_ signature --- the expert's output
does not lie near any low-dimensional vocabulary subspace --- but it is not, on its own, evidence
of _polysemanticity_: an under-explained readout could equally reflect an output that is simply
off the unembedding manifold. We therefore test polysemanticity directly in @sec:polysemy, by
asking whether an expert's top atoms span many _unrelated_ token families; they do. Both signatures
match the MoE interpretability literature, which reports that experts pack features in
superposition @lecomte2025sparsity and that single experts under-determine model behavior
@monosemanticpaths2026 @illusionspecialization2026. Nevertheless, a substantial minority of
experts exhibit clear semantic specialization, mostly in the later layers. @tab:experts shows
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

Specialists tend to cluster in the later layers (L12--L15), consistent with the view that
deeper layers encode more abstract semantic content. Early layers (L00--L04) exhibit lower EVR
values and less coherent token lists, suggesting they operate on lower-level distributional
features.

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
    [Logit lens (mean direction)], [0.0010], [0.0025], [0.0069],
    [SOMP (variance basis)],       [0.0020], [0.0056], [0.0146],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Cumulative EVR averaged over all 1,024 experts at decomposition depths 1, 3, and 10, for the
mean-projection logit lens versus SOMP, computed on a pile (10k-document) extraction with both
readouts on the same activations. SOMP
explains $approx 2.1 times$ more activation variance at every depth. The mean top-10 token
overlap between the two readouts (Jaccard) is only 0.043, so the two methods largely disagree
on which tokens characterize an expert.
  ],
) <tab:lens>

@tab:lens shows two things. First, the absolute EVR is small for both methods --- even ten
atoms explain under 2% of an expert's activation variance --- a direct, quantitative signature
of polysemanticity: an expert's output does not lie near any low-dimensional vocabulary
subspace @lecomte2025sparsity. Second, SOMP consistently captures $approx 2.1 times$ the
variance of the logit lens, and the two readouts share almost no top tokens (Jaccard $0.043$).
A single mean-direction ranking therefore systematically under-reads an expert: it is biased
toward the high-norm mean rather than the directions along which the expert actually varies.
This is the per-expert analogue of the cross-layer finding that semantics in MoEs live in
distributed structure rather than any single component @monosemanticpaths2026, and it is the
empirical justification for preferring a sparse multi-atom basis over a one-shot logit lens.

== Polysemanticity, Measured Directly <sec:polysemy>

Low EVR establishes that experts live in superposition, but polysemanticity is the stronger,
_semantic_ claim that a single expert mixes many _unrelated_ concepts. We test it directly in the
model's own readout geometry: every SOMP atom is a row of the unembedding, so two atoms are
"related" when their unembedding rows are aligned (cosine $gt.eq 0.4$) and unrelated when
near-orthogonal. For each expert we cluster its top-30 atoms (single-linkage at that threshold)
and measure the _largest-family share_ --- the fraction of atoms falling in the biggest cluster.
A monosemantic readout is dominated by one family (share near 1); a polysemantic grab-bag splits
into many singletons (share near $1\/30$).

The median expert has a largest-family share of just *0.03*: its biggest coherent token group is
about one atom out of thirty. A genuine single-topic lexicon, by contrast --- the `numbers`
concept words, clustered identically --- sits at *0.37*, an order of magnitude tighter, and only
*4%* of experts have a coherent core of four or more related atoms. The effect is not an artifact
of SOMP's atom decorrelation: a truly single-topic expert yields a clustered atom set (the sharpest
expert, British/formal spelling at EVR 0.45, does retain an `amongst`/`among`/`whilst`/`realised`
core), whereas the typical expert does not. Strikingly, the single _highest_-EVR expert
(`L00 E06`, which fires on almost every prompt) is a complete grab-bag (`this`, `The`, `Kindle`,
`Sutton`, `History`, ...) --- high EVR is not monosemanticity. The picture is consistent: real
specialization is a thin on-theme layer over a polysemantic core. The measure is reproducible via
`scripts/atom_polysemanticity.py`.

== From Association to Causation: A Localizability Gradient <sec:causal>

The pursuit results above are correlational. We now test, on OLMoE, _which_ experts causally
drive a concept and whether acting on them removes it (@sec:causal-methods), running the same
selectors~$times$~interventions pipeline on three concepts of decreasing lexical sharpness:
`countries`, `numbers`, and toxicity (`offensive`).

#emph[Data and protocol.] All causal experiments use RealToxicityPrompts
@gehman2020realtoxicityprompts, split by each prompt's own toxicity score into eliciting (high) and
neutral (low) halves and then into disjoint train (identify) and test (score) sets. The toxicity
run identifies on $n_"train" = 100$ and scores on $n_"test" = 64$; the lexical-concept runs use the
smaller split written by the per-concept pursuit ($n_"train" = 60$, $n_"test" = 40$ for
`countries`/`numbers`). The SOMP selector is the concept-restricted Expert-Pursuit ranking
(dictionary restricted to the concept lexicon, ranked by EVR\@10); the gate-AtP selector is the
grid of @eq:atp computed on the train eliciting prompts. Every number below is on held-out test
prompts.

The headline is that causal _controllability_ is a *gradient* --- countries is sharply
localizable, numbers only weakly, toxicity not at all --- and that within every concept the
separating signal is *influence, not necessity*: nudging the right experts removes the concept, but
knocking them out never does. The correlational SOMP selector, finally, is never cleanly causal:
where it "works" it does so only by degrading the text.

=== Causal Localization Is a Per-Concept Gradient

Gate-AtP (@eq:atp) scores every $(l,e)$ from a single backward pass, giving one signed
$16 times 64$ map per concept. The three maps differ qualitatively (@fig:atp-grids): for
`countries` the attribution concentrates in a handful of strong late-layer experts --- a sharp,
$approx 1%$-of-experts handle; for `numbers` the signal exists but is spread thinly across many
experts (distributed, leaky); for toxicity it is diffuse and low-magnitude everywhere, with no
sparse lever. This _localizability gradient_ is the organizing finding of the section, and the
interventions below trace it: each concept is exactly as controllable as its map is concentrated.

#figure(
  image("../figures/grid_atp_concepts.png", width: 100%),
  caption: [
Gate-AtP attribution maps ($16$ layers $times$ $64$ experts) for `countries`, `numbers`, and
`offensive` (toxicity). Colour is the signed first-order gate-ablation effect (@eq:atp); red
promotes the concept, blue suppresses it. The maps grow visibly more diffuse left-to-right:
`countries` concentrates on a few late-layer experts, `numbers` spreads thinly, and toxicity has
no sparse lever --- the localizability gradient that the interventions trace.
  ],
) <fig:atp-grids>

For toxicity specifically, the exact patching grid (@eq:patch) confirms the diffuse picture and
adds a twist invisible to a vocabulary readout: the causally important experts are *not* confined
to the late layers where the pursuit specialists live. @tab:patch lists the nine with the largest
absolute effect --- they span layer~1 to layer~15, roughly half are *suppressors* (ablating them
_raises_ toxicity), the largest single effect is only $approx 0.04$ probe units, and no expert
dominates. Pursuit's high-EVR specialists cluster late only because projecting _early_-layer
activations onto the unembedding is ill-posed; the causal grid has no such bias. The tiny,
sign-mixed, depth-spread effects are the first quantitative hint of the top-$k$ redundancy that
the interventions confirm.

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
Experts with the largest absolute causal effect (@eq:patch) on the toxic-logit probe. Positive =
the expert promotes toxicity (ablation lowers the score); negative = suppressor. The effects span
all depths, unlike the late-layer pursuit specialists, and are uniformly small.
  ],
) <tab:patch>

=== Faithfulness: gate-AtP Recovers the Gold Grid Cheaply --- so We Can Drop It

The patching grid is the causal ground truth but costs one forward pass per routed expert; as a
_selector_ it is impractical. @tab:faith shows we do not need it: gate-AtP --- a single backward
pass --- tracks it moderately overall (pooled $r approx 0.69$) and *highly* in the late layers
($r approx 0.91$--$0.96$ for L12--L15), degrading only in the early layers ($r approx
0.30$--$0.49$) where per-expert effects are near zero and the first-order gradient is noisiest.
Because the controllable signal lives in exactly the late layers where AtP is faithful, every
intervention below is driven by the cheap AtP grid and patching is used only as this yardstick.

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
    [gate-AtP (gradient)],   [1 backward pass],         [$+0.69$], [$+0.93$],
    [patching (gold)],       [1 forward / expert],      [---],     [---],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
One-off faithfulness check of gate-AtP against the exact patching grid (Pearson $r$ over scored
experts). The gradient proxy is moderately faithful pooled and highly faithful in the late layers
--- where the controllable signal lives --- so it stands in for the $approx 64 times$ more
expensive grid, which we then drop. (Patching is the ground truth, so its self-$r$ is trivially~1.)
  ],
) <tab:faith>

#figure(
  image("../figures/patching_faithfulness.png", width: 100%),
  caption: [
Faithfulness of gate-AtP against the exact patching grid on toxicity. _Left:_ the exhaustive
patching effect map. _Centre:_ gate-AtP vs patching per expert (each point an $(l,e)$); the cloud
tightens toward the diagonal in the late layers. _Right:_ per-layer Pearson $r$, rising from
$approx 0.30$ early to $approx 0.96$ in the last layer --- AtP is most faithful exactly where the
controllable signal lives.
  ],
) <fig:faithfulness>

=== Influence vs. Necessity

With the cheap selector validated we ask the decisive question: does acting on a concept's AtP
experts remove the concept? The answer splits cleanly by _mode_. *Influence* --- localized
steering, adding $alpha bold(v)$ to the selected experts (@sec:interventions) --- works, and
sharply for the localizable concepts. On `countries`, steering the top-$k=15$ AtP experts drives
the country word-fraction from a $0.60$ baseline to $0.17$ at $alpha=-3$ and to $0.03$ at
$alpha=-5$, all while distinct-1 stays $approx 0.83$ (coherent removal, not degradation).
`numbers` confirms the gradient: it never gets a clean lever, needing both $k=40$ experts _and_
$alpha=-5$ to reach $0.43$ (@tab:influence, @fig:steer). Toxicity, at the bottom of the gradient,
does not move under expert steering at all.

*Necessity* --- knockout --- fails everywhere. Zeroing even the top $10%$ of all 1,024 experts
($p_90$, 103 experts) leaves the country word-fraction at $0.33$ for the AtP set vs $0.47$ for a
layer-matched random set (@fig:knockout): a real but small gap that never makes the concept
disappear, and `numbers` shows no AtP-vs-random gap at all. With 8 experts active per token the
model simply routes around any sparse set --- *top-$k$ redundancy*.

This is the central negative result, and it is worth stating precisely against Head Pursuit
@basile2025headpursuit, where editing as few as $approx 1%$ of the SOMP-identified _heads_ reliably
suppresses or enhances a concept. Two things differ in our setting --- the _selector_ (SOMP vs
gate-AtP) and the _operator_ (Head Pursuit rescales a component's whole contribution, knockout
zeroes a gate) --- so to isolate the cause we re-ran Head Pursuit's _own_ operator, an $alpha = -1$
output rescale, on the OLMoE experts (@tab:hp). On `countries` it moves the word-fraction only
$0.60 -> 0.47$ on the SOMP experts --- _exactly_ the layer-matched random number ($0.47$) --- and
$0.60 -> 0.43$ on the AtP experts; it never isolates a causal set. The descriptive SOMP story thus
survives the head$->$expert transfer, but the causal-_localization_ story does not, and this holds
under Head Pursuit's intervention as well as ours: the obstruction is the redundancy of $8$-of-$64$
routing, not the choice of operator. Only our per-expert diff-of-means steering separates the causal
set at all ($0.03$ for AtP vs $0.37$ for random). The right reading is that the AtP experts have
causal _influence_ over the concept without being individually _necessary_ for it.

#figure(
  table(
    columns: (1.4fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Intervention* (`countries`, base wf $0.60$)], [*SOMP*], [*AtP*], [*random*], [*distinct-1*],
      table.hline(stroke: 0.5pt),
    ),
    [Head Pursuit rescale ($alpha = -1$)],       [$0.47$], [$0.43$], [$0.47$], [$0.79$--$0.84$],
    [knockout ($alpha = 0$)],                    [$0.40$], [$0.43$], [$0.47$], [$approx 0.82$],
    [diff-of-means steer ($alpha = -5$, ours)],  [$0.30$ #text(fill: rgb("#b00"))[(d1 $0.66$)]], [$bold(0.03)$], [$0.37$], [$0.83$--$0.85$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Head Pursuit's intervention vs ours on `countries` (word-fraction, lower = more removed; base
$0.60$, $n_"test" = 30$). Head Pursuit's own $alpha = -1$ output rescale ties the layer-matched
random control on every selector --- it does not isolate a causal set on experts --- and the SOMP
set only "moves" under aggressive diff-of-means steering by collapsing coherence (distinct-1
$0.66$). Only diff-of-means steering of the *gate-AtP* experts removes the concept cleanly ($0.03$
vs $0.37$ random). The descriptive head$->$expert transfer survives; the causal-localization
transfer does not, under Head Pursuit's operator as well as ours.
  ],
) <tab:hp>

#figure(
  image("../figures/threshold_knockout.png", width: 85%),
  caption: [
Necessity test on `countries`: country word-fraction as a function of the number of knocked-out
experts, for the gate-AtP set vs a layer-matched random set. Even at $10%$ of all experts ($103$)
the concept never disappears and the AtP-vs-random gap stays small --- with 8 experts firing per
token the model routes around any sparse knockout (top-$k$ redundancy).
  ],
) <fig:knockout>

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Concept*], [*base*], [$alpha = -1$], [$alpha = -3$], [$alpha = -5$], [*distinct-1*],
      table.hline(stroke: 0.5pt),
    ),
    [`countries` ($k = 15$)], [$0.60$], [$0.43$], [$0.17$], [$bold(0.03)$], [$approx 0.83$],
    [`numbers` ($k = 40$)],   [$0.80$], [$0.80$], [$0.80$], [$0.43$],       [healthy],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Influence test: concept word-fraction (lower = less concept) under localized steering of the
top-$k$ gate-AtP experts at increasing strength $alpha$. `countries` has a sharp, coherent lever
(0.60$->$0.03); `numbers` is leaky even at $k = 40$. Distinct-1 confirms the drops are clean
removal, not degraded text.
  ],
) <tab:influence>

#figure(
  image("../figures/steering_sweep.png", width: 100%),
  caption: [
Influence test: concept word-fraction under localized expert-output steering as the strength
$alpha$ becomes more negative, for `countries` (left) and `numbers` (right) at several $k$.
`countries` saturates at $k approx 15$ ($approx 1%$ of experts) and falls to $0.03$; `numbers`
needs $k = 40$ and $alpha = -5$ just to reach $0.43$ --- the distributed concept never gets a
clean lever.
  ],
) <fig:steer>

The numbers are easiest to read on the generations themselves (@fig:generation): steering the gate-AtP
experts removes country names while the continuation stays fluent and on-topic, whereas knocking the
same experts out leaves the country names in place. Fluency is read two ways --- the *distinct-1*
unique-unigram ratio (held at $approx 0.79$ here, i.e. no degeneration) and the continuations
themselves, which stay grammatical under steering.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Intervention*], [*Sample continuation, held-out `countries` prompt*],
      table.hline(stroke: 0.5pt),
    ),
    [baseline],                   ["...the 2022 Olympics in Beijing, China ... held every four years..."],
    [AtP-steer ($alpha {=} {-}5$)], ["...a place called Tokyo ... you can just go to a different country and use it there..."],
    [AtP-knockout],               ["...the 2024 Olympics in Paris, France ... held every four years..."],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Generation under intervention (greedy decoding, held-out `countries` prompts). Steering the gate-AtP
experts strips country names while keeping the text fluent (country word-fraction $0.68 -> 0.05$,
distinct-1 $approx 0.79$); knocking the same experts out leaves the names intact (word-fraction
$0.5$). The steered text is grammatical --- it simply stops naming countries --- which is how
"clean removal" differs from the degeneration a distinct-1 collapse would flag.
  ],
) <fig:generation>

=== Specificity, and Why SOMP Only "Works" by Breaking Generation

A genuine localization must be *specific*: steering down one concept with its own AtP set should
spare the other concept, fluently. It does (@tab:specificity). Steering the `countries`-AtP
experts collapses the country word-fraction $0.60 -> 0.03$ while `numbers` survives at $0.73$ and
distinct-1 holds at $approx 0.79$; the symmetric `numbers`-AtP edit suppresses numbers
($0.80 -> 0.37$) and spares countries ($0.53$). Each concept's causal experts suppress _their
own_ concept and leave the other near baseline.

The correlational SOMP selector behaves entirely differently, and this is the sharpest statement
of its causal failure. SOMP-selected experts only "reduce" a concept under aggressive steering
($alpha = -10$), and when they do, *distinct-1 collapses to $0.27$--$0.59$* --- the model is
degrading the text into repetition, not removing the concept. SOMP thus passes a naive
propensity-drop test for the wrong reason. The coherence guard is what exposes it: under the
identical steering machinery, the gate-AtP set removes the concept cleanly and SOMP does not ---
token association is not causal responsibility.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Intervention*], [*country wf*], [*number wf*], [*distinct-1*],
      table.hline(stroke: 0.5pt),
    ),
    [baseline],                                  [$0.60$], [$0.80$], [$0.72$--$0.79$],
    [`countries`-AtP steer ($alpha = -5$)],      [$bold(0.03)$], [$0.73$ #text(fill: gray)[(spared)]], [$0.76$--$0.79$],
    [`numbers`-AtP steer ($alpha = -10$)],       [$0.53$ #text(fill: gray)[(spared)]], [$bold(0.37)$], [$0.79$],
    [SOMP steer ($alpha = -10$)],                [reduces], [reduces], [#text(fill: rgb("#b00"))[0.27--0.59]],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Specificity test (word-fraction, lower = more removed). Each concept's gate-AtP experts suppress
their own concept and spare the other with coherence intact; the correlational SOMP set only lowers
the metric by collapsing distinct-1, i.e. by degrading generation rather than removing the concept.
  ],
) <tab:specificity>

=== The Toxicity Tail: No Expert Lever At All

Toxicity sits at the bottom of the gradient, and the expert-level interventions confirm it has
*no usable expert handle*. @tab:intervene reports the held-out toxic-logit propensity under
each expert intervention relative to a $+2.07$ baseline ($n_"train" = 100$, $n_"test" = 64$).
Knockout is near-inert: the causal AtP set retains the right _ordering_ (AtP $-0.10$ beats SOMP
$-0.04$ and random $-0.03$, exactly as the faithfulness check predicts), but no set meaningfully
moves the metric --- top-$k$ redundancy again. Expert-output steering is the strongest arm and
still barely works: even at $alpha = -10$ the causal set drops toxicity only $-0.33$ ($approx 16%$)
and drags the _neutral_ prompts down almost as far ($-0.20$), so the specificity margin is a thin
$+0.13$ --- better than random's $+0.07$, but nowhere near the clean removal `countries` showed
(@tab:specificity). Unlike the lexical concepts, toxicity is *semantic and fully distributed*: it
does not concentrate on any sparse expert set, so no expert-level edit removes it cleanly. This is
the honest endpoint of the gradient --- some behaviors simply are not expert-localizable, and an
expert-only toolkit should report that rather than reach for a residual-stream edit.

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Intervention*], [*Toxic prop.*], [*$Delta$ vs base*], [*Neutral $Delta$*], [*distinct-1*],
      table.hline(stroke: 0.5pt),
    ),
    [baseline],                       [$+2.07$], [---],     [---],     [$0.83$],
    [AtP-knockout (top 15)],          [$+1.96$], [$-0.10$], [$-0.06$], [$0.82$],
    [SOMP-knockout],                  [$+2.03$], [$-0.04$], [$-0.05$], [$0.81$],
    [random-knockout],                [$+2.04$], [$-0.03$], [$-0.02$], [$0.84$],
    [AtP esteer ($alpha = -10$)],     [$bold(+1.74)$], [$bold(-0.33)$], [$-0.20$], [$0.82$],
    [SOMP esteer ($alpha = -10$)],    [$+1.93$], [$-0.14$], [$-0.07$], [$0.83$],
    [random esteer ($alpha = -10$)],  [$+1.92$], [$-0.15$], [$-0.08$], [$0.84$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Expert-level interventions on toxicity (held-out, $n_"test" = 64$; lower propensity = less toxic).
$Delta$ is the eliciting reduction, *Neutral $Delta$* the collateral drop, *distinct-1* the
coherence guard. Knockout is near-inert for every selector; expert-output steering is the strongest
arm but only weakly and non-specifically (AtP's $-0.33$ comes with a $-0.20$ neutral drop), so
toxicity has no clean expert lever --- the diffuse bottom of the localizability gradient.
  ],
) <tab:intervene>

Taken together the three concepts trace one gradient --- `countries` (sharp, $approx 1%$ handle)
$>$ `numbers` (distributed, leaky) $>$ toxicity (redundant, direction-only) --- and one recurring
lesson: causal _influence_ is recoverable from a cheap gate gradient, but expert _necessity_ is
an illusion of redundant routing, and the only selector that survives a coherence-aware test is
the causal one. @tab:gradient collects the headline numbers for all three concepts in one place.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Concept*], [*AtP map*], [*Best steer (base$->$best)*], [*Knockout (AtP vs rand)*], [*Verdict*],
      table.hline(stroke: 0.5pt),
    ),
    [`countries`], [sharp, $approx 1%$ late], [wf $0.60 -> bold(0.03)$, $k{=}15,alpha{=}{-}5$, d1~$0.83$], [$0.33$ vs $0.47$ @ $p_90$], [localizable],
    [`numbers`],   [distributed],            [wf $0.80 -> 0.43$, $k{=}40,alpha{=}{-}5$],            [no gap],                  [leaky],
    [toxicity],    [diffuse],                [prop $+2.07 -> +1.74$ ($-16%$), $alpha{=}{-}10$],     [$-0.10$ vs $-0.03$],      [not localizable],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
The localizability gradient at a glance. Each concept's best localized-steering result, its
knockout effect (AtP vs layer-matched random), and the qualitative shape of its gate-AtP map.
Word-fraction (wf) is the metric for the lexical concepts; toxicity uses the concept-logit
propensity. Controllability tracks how concentrated the AtP map is: sharp $->$ clean lever,
diffuse $->$ no expert handle.
  ],
) <tab:gradient>

=== Stress-Testing the Redundancy: Sufficiency and Co-Firing Groups

Two further checks confirm the toxicity knockout is redundancy-bound, not mis-targeted. A
*sufficiency curve* knocks out the top-$j$ experts of each selector for growing $j$ and scores the
held-out toxic probe (@tab:sufficiency; throughout this subsection, following the table convention,
*positive = less toxic*). The gate-AtP set lowers toxicity _monotonically_ at the reported budgets
but the reduction only reaches $+0.21$ at $j = 103$ (10% of all experts), and it is the _only_
selector whose effect is statistically resolved (95% bootstrap CI $[+0.11, +0.30]$ at $j = 103$);
SOMP and random straddle zero at every budget, and SOMP even _raises_ toxicity at large $j$.
Distinct-1 holds at $approx 0.83$ throughout, so these are coherent ablations, not degradation. A
*co-firing group ablation* agrees: padding the 15 AtP experts with their four top co-firing
neighbours each (69 experts) deepens the reduction to $+0.16$ (probe $2.07 -> 1.91$) versus only
$+0.02$ for a size-matched random group --- better, and causal, but still nowhere near removal. Sparse knockout is defeated by redundant routing even
when the set is causal and enlarged along co-firing structure; only direction-level steering removes
the concept.

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
Sufficiency curve: reduction in the toxic-logit probe (baseline $+2.07$; positive = less toxic) as
the top-$j$ experts of each selector are knocked out, held-out $n_"test" = 64$, distinct-1
$approx 0.83$ throughout. Only gate-AtP falls monotonically, and only its $j = 103$ point is
bootstrap-significant (95% CI $[+0.11, +0.30]$); SOMP and random straddle zero. Knocking out even
$10%$ of all experts leaves most toxicity intact --- top-$k$ redundancy.
  ],
) <tab:sufficiency>

== Cross-Model Check: GPT-OSS-20B

We re-ran the pipeline on a second, larger model --- `openai/gpt-oss-20b`, a 24-layer sparse MoE
with 32 local experts per layer, top-4 routing, and hidden size 2880 @openai2025gptoss --- to test
whether the central claims are architecture-specific. They are not. Three findings replicate. (1)
*Polysemanticity*: the median per-expert EVR is $0.042$ (pile-10k extraction), essentially
identical to OLMoE's $0.042$ on the same pile-10k extraction, so a single readout under-reads experts in both models.
Qualitatively, though, gpt-oss's experts are _less lexically interpretable_: even on TriviaQA its
highest-EVR experts are dominated by multilingual fragments, code, and punctuation rather than the
clean semantic categories OLMoE shows (@tab:experts), consistent with its 200k multilingual vocabulary
and coarser top-4 routing --- so the clean per-expert _token_ summaries are an OLMoE phenomenon even
though the low-EVR polysemanticity is shared. (2) *AtP
faithfulness*: gate-AtP tracks the exhaustive patching grid even more closely here than on OLMoE
(pooled Pearson $r approx 0.77$), reinforcing that the cheap one-pass gradient stands in for the
expensive sweep across architectures --- the reason patching is not part of the pipeline. (3)
*Knockout redundancy*: on a held-out toxicity split the causal gate-AtP set again separates from
random, but no selector cleanly removes toxicity --- top-8 knockout lowers the probe by
$approx 0.4$ for AtP versus $approx 0$ for random, yet with coherence loss (distinct-1
$approx 0.65$) and a sizeable collateral drop on the _neutral_ prompts ($approx 0.2$), so the
effect is neither clean nor specific. With only $n_"train" = 50$ / $n_"test" = 30$ prompts the per-selector intervention picture
is noisy, so we read GPT-OSS as a *replication of the descriptive and faithfulness claims* rather
than a second intervention study.
