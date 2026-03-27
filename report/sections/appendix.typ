= Appendix <appendix>

== OLMoE Architecture

OLMoE-1B-7B-Instruct @muennighoff2024olmoe is a sparse MoE language model with 1B active
parameters out of 7B total. Its architecture has the following key properties relevant to
Expert Pursuit:

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Property*], [*Value*],
      table.hline(stroke: 0.5pt),
    ),
    [Layers ($L$)],             [16],
    [Experts per layer],        [64],
    [Active experts (top-$k$)], [8],
    [Model dimension ($d$)],    [2048],
    [Vocabulary size ($v$)],    [~50,000],
    [Expert FFN hidden dim],    [1024],
    table.hline(stroke: 0.8pt),
  ),
  caption: [OLMoE-1B-7B-Instruct architecture summary.],
) <tab:olmoe-arch>

Each MoE layer replaces the standard FFN with 64 expert FFNs and a learned token-choice
router. The router applies a softmax over expert logits and selects the top-8 experts per
token. The gated output of expert $e$ for token $bold(x)$ is:

$ bold(c)_e (bold(x)) = g_e (bold(x)) dot f_e (bold(x)) $

where $g_e$ is the scalar gating weight and $f_e (bold(x)) in RR^d$ is the expert FFN output.
Expert Pursuit operates on $bold(c)_e$ averaged over documents.

== GPT-OSS Notes

`openai/gpt-oss-20b` is also supported by the extraction and pursuit code. It is a 24-layer
MoE model with 32 local experts per layer, 4 experts routed per token, and hidden size 2880
@openai2025gptoss. We keep the main report centered on OLMoE because that is the model for
which the current results were generated.

== Concept Word Lists <app:concepts>

The concept-restricted pursuit mode restricts the SOMP dictionary to token IDs
corresponding to predefined word lists. The three lists used in this work are defined in
`src/concepts.py`.

*Numbers.* Digit strings (0--9, 10, 100, 1000) and English number words:
_zero, one, two, ..., nineteen, twenty, thirty, ..., ninety, hundred, thousand, million,
billion_, plus ordinals _first, second, third, fourth, fifth_ and quantifiers _half,
quarter, double, triple, single, pair, dozen_.

*Countries.* 50 country names covering major world regions, including _France, Germany,
Japan, Brazil, Nigeria, India, Australia_, etc. Multi-word names such as _United Kingdom_
and _South Korea_ are included as single entries; the tokenizer may split these into
multiple tokens, all of which are included in the restricted dictionary.

*Offensive.* A list of 67 words associated with harmful or sensitive content, including
_violence, hate, terrorism, racism_, etc. This list was used to probe whether any experts
specialize in harmful content; none of the top-ranked experts under this concept showed
coherent specialization on TriviaQA.

== SOMP Algorithm <app:somp>

Algorithm~1 gives the SOMP procedure used by Expert Pursuit. The dictionary $bold(D)$ is
L2-normalized row-wise before the run; rows correspond to vocabulary tokens. At each step,
the atom with the highest sum of squared correlations across all $n$ documents is selected.
Coefficients are refitted by least-squares on the growing support set, and the residual is
updated. The EVR at step $t$ is:

$ "EVR"^t = 1 - frac(norm(bold(R)^t)_F^2, norm(bold(H))_F^2) $

where $bold(R)^t$ is the residual after $t$ steps and $bold(H)$ is the centered
activation matrix.
