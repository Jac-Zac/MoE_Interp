= Results <sec:results>

We ran Expert Pursuit on 50,000 TriviaQA questions using OLMoE-1B-7B-Instruct, capturing
last-token gated outputs for all 16 layers $times$ 64 experts. Of the 1,024 experts, 393 had
sufficient activations (at least 5 routed documents) to analyze. These results are a snapshot
of the current last-token pipeline.

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

== GPT-OSS Support

The codebase also supports `openai/gpt-oss-20b` as a second target model. GPT-OSS is a
24-layer sparse MoE model with 32 local experts per layer, 4 experts routed per token, and
hidden size 2880 @openai2025gptoss. We do not report GPT-OSS results here, but the same
extraction and pursuit pipeline can be applied to it for future comparison.
