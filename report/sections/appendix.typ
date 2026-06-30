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
`src/moe_interp/pursuit/concepts.py`.

*Numbers.* Digit strings (0--9, 10, 100, 1000) and English number words:
_zero, one, two, ..., nineteen, twenty, thirty, ..., ninety, hundred, thousand, million,
billion_, plus ordinals _first, second, third, fourth, fifth_ and quantifiers _half,
quarter, double, triple, single, pair, dozen_.

*Countries.* 50 country names covering major world regions, including _France, Germany,
Japan, Brazil, Nigeria, India, Australia_, etc. Multi-word names such as _United Kingdom_
and _South Korea_ are included as single entries; the tokenizer may split these into
multiple tokens, all of which are included in the restricted dictionary.

*Offensive.* A list of 66 words associated with harmful or sensitive content, including
_hate, terrorism, violent, racist_, etc. This list was used to probe whether any experts
specialize in harmful content; none of the top-ranked experts under this concept showed
coherent specialization on TriviaQA.

== SOMP Algorithm <app:somp>

@app:somp-alg gives the SOMP procedure used by Expert Pursuit. The dictionary $bold(D)$ is
L2-normalized row-wise before the run; rows correspond to vocabulary tokens. At each step,
the atom with the highest sum of *absolute* correlations across all $n$ documents is selected
(the $ell_1$ criterion of Head Pursuit @basile2025headpursuit, $p^t = arg max_j norm(bold(D)_j bold(R)^(t top))_1$).
Coefficients are refitted by least-squares on the growing support set, and the residual is
updated. The EVR at step $t$ is

$ "EVR"^t = 1 - frac(norm(bold(R)^t)_F^2, norm(bold(H))_F^2) $

where $bold(R)^t$ is the residual after $t$ steps and $bold(H)$ is the centered
activation matrix.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [SOMP decomposition for one expert (default $ell_1$ criterion).],
  block(width: 100%, inset: 8pt, stroke: 0.5pt, radius: 3pt, align(left, [
    #set text(size: 9pt)
```
Input:  centered data H (n×d); L2-normalized dictionary D (v×d); steps T
Output: support S (ranked tokens); EVR curve
1  R ← H ;  S ← ∅
2  for t = 1 … T:
3      c_j ← Σ_i |⟨R_i, D_j⟩|   for every atom j ∉ S        # ℓ1 correlation
4      p   ← argmax_{j∉S} c_j ;  S ← S ∪ {p}
5      W   ← argmin_W ‖H − W·D_S‖_F²                         # least-squares refit
6      R   ← H − W·D_S
7      EVR_t ← 1 − ‖R‖_F² / ‖H‖_F²
8  return S ordered by ‖W‖ ,  {EVR_t}
```
  ])),
) <app:somp-alg>

== gate-AtP Algorithm <app:atp>

@app:atp-alg gives the causal selector of @sec:selectors. It scores every $(l, e)$ from a
single forward--backward pass, the first-order approximation of the $approx 1024$ ablation
forward passes that exhaustive patching (@eq:patch) would cost. The only adaptation to MoE
experts is the *node we differentiate*: the fused-experts kernel never materializes the
per-expert hidden neurons, so the finest differentiable per-expert handle is the router gate
$g_e$. Because the block output $sum_e g_e f_e$ is *linear* in $g_e$, the gate carries no
node-level saturation, and AtP's only error is the downstream nonlinearity below the residual
stream --- which is why it tracks the gold patching grid closely in the late layers (@sec:results).

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [gate-AtP: per-$(l, e)$ causal attribution in one backward pass.],
  block(width: 100%, inset: 8pt, stroke: 0.5pt, radius: 3pt, align(left, [
    #set text(size: 9pt)
```
Input:  prompts; concept token set C; metric s_C (concept-logit probe)
Output: attribution grid AtP ∈ R^{L×E}
1  AtP[l, e] ← 0
2  forward pass; at each layer l read the fused-experts boundary (h, idx, g);
                  mark the gates g differentiable
3  L ← Σ_{prompt} s_C(z_last)                              # metric over the batch
4  backward pass → ∂L/∂g for every layer l
5  for each layer l, for each routed slot (token, expert e in idx):
6      AtP[l, e] += g · ∂L/∂g                              # contribution = −Δmetric on g→0
7  return AtP ;  rank experts by signed AtP
```
  ])),
) <app:atp-alg>
