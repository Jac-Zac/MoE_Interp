= Results

We ran Expert Pursuit on 50,000 TriviaQA questions using OLMoE-1B-7B-Instruct, capturing
last-token gated outputs for all 16 layers $times$ 64 experts. Of the 1,024 experts, 393 had
sufficient activations (at least 5 routed documents) to analyze. These results are a snapshot
of the current last-token pipeline.

== Expert Specialization

Final EVR values (after $T = 25$ SOMP iterations) range from 0.021 to 0.248, with a median
of 0.038. The low median reflects that most experts are polysemantic --- no small set of
vocabulary directions captures most of their variance. Nevertheless, a substantial minority of
experts exhibit clear semantic specialization. @tab:experts shows representative examples
across several categories identified by full-dictionary pursuit.

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
    [L14 E02], [Names],            [David, Steve, John, Andrew, George, Bill, King],   [0.048],
    [L15 E16], [Names],            [Ryan, Smith, Bobby, Oliver, Shannon, Charles],     [0.044],
    [L15 E38], [Biology],          [protein, human, digestive, blood, plant, chemical],[0.054],
    [L15 E59], [Entertainment],    [music, comic, debut, screen, thriller, play],      [0.050],
    [L14 E08], [Food],             [fruit, olive, pot, chicken, food],                 [0.036],
    [L14 E51], [Religion],         [beloved, child, brother, pray, honey, friends],    [0.077],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Selected experts identified by full-dictionary pursuit on 50,000 TriviaQA documents. EVR is
the cumulative explained variance ratio after 25 SOMP iterations.
  ],
) <tab:experts>

Specialists tend to cluster in the later layers (L12--L15), consistent with the view that
deeper layers encode more abstract semantic content. Early layers (L00--L04) exhibit lower EVR
values and less coherent token lists, suggesting they operate on lower-level distributional
features.

== Concept-Restricted Pursuit: Numbers

To validate the concept-restricted mode, we ran Expert Pursuit with the dictionary restricted
to the `numbers` word list (digit tokens plus English number words; see `src/concepts.py`).
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
    [L15 E03], [10, 100, 4, 8, six],        [0.057],
    [L14 E37], [9, 3, 100, 4, 7],           [0.054],
    [L15 E48], [4, 10, three, half, een],   [0.045],
    [L13 E55], [third, two, 8, 6, 1],       [0.043],
    [L11 E43], [3, 4, one, ten, seven],     [0.042],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
Top 5 experts ranked by EVR under concept-restricted pursuit with the `numbers` dictionary
(run on a smaller local sample of 144 documents).
  ],
) <tab:numbers>

The ranking agrees with the full-dictionary results: L15 E03 and L14 E37 rank first and
second in both modes, and their unrestricted token lists consist almost entirely of numerals
and year tokens (@tab:experts). Concept-restricted pursuit thus serves as a focused probe that
confirms and quantifies specialization identified by the full-dictionary run.

== GPT-OSS Support

The codebase also supports `openai/gpt-oss-20b` as a second target model. GPT-OSS is a
24-layer sparse MoE model with 32 local experts per layer, 4 experts routed per token, and
hidden size 2880 @openai2025gptoss. We do not report GPT-OSS results here, but the same
extraction and pursuit pipeline can be applied to it for future comparison.
