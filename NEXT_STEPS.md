# Next Steps

This is a ranked roadmap for Expert Pursuit. Keep tests tensor-only; only model
runs should live behind CLI/notebook workflows.


### Mean unembedding projection baseline

For each expert, compute `topk(mean(activations) @ W_U.T)` as a simpler baseline alongside SOMP. This is not a logit lens (which is per-sentence) but a bulk mean-projection — useful to check how much SOMP adds over a naïve aggregate. The problem with this is the assumpton of *monosematicity of experts*.

For qualitative per-expert validation (actual logit lens style): after SOMP identifies an interesting expert, craft a relevant prompt, run a single forward pass, and decode `topk(h_expert @ W_U.T)` for that expert on that prompt.

### Per-token frequency analysis

The HDF5 files already store `token_ids` alongside activations. For each expert, group rows by `token_id`, count occurrences, and rank. Cross-reference with SOMP top tokens: if expert X both routes most on math tokens AND SOMP gives math atoms, that is convergent validation from two independent angles.

### PCA singular value spectrum

For each expert's activation matrix `A` (already in HDF5), compute `torch.linalg.svd(A)` and record `σ₁ / Σσᵢ` (fractional variance in the first direction). Plot as a heatmap over layers × experts. Classifies experts as monosemantic (`σ₁` dominant) vs polysemantic (flat spectrum). Add as a new analysis mode in `src/moe_interp/analysis/`.


## 0. First Fixes

- Add a short `method_notes.md` or README section defining exactly what is stored:
  routed expert output, already multiplied by routing weight, after component RMSNorm.
- Rename "PCA singular value spectrum" to "expert output spectrum" unless centered PCA
  is used. Report both:
  - `mean_projection`: top tokens from `mean(A) @ W_U.T`
  - `pc1_evr`: `s[0] ** 2 / (s ** 2).sum()` from centered `A`
- Treat current HDF5 token counts as last-token routing counts only. They are useful,
  but they are not a replacement for full content-token routing.

## 1. Existing-HDF5 Analyses

Build these first because they need no new capture.

### 1a. Mean Projection Baseline

For each expert, compute `topk(mean(A) @ W_U.T)`. Save beside SOMP results.

Why: quick sanity check for whether SOMP adds value over a simple aggregate logit-lens
style projection.

### 1b. Expert Output Spectrum

Add `src/moe_interp/analysis/spectrum.py`.

For each expert activation matrix `A`:
- Center `A` before SVD for PCA: `A_centered = A - A.mean(0)`.
- Return singular values, `pc1_evr`, effective rank, and top tokens for the first 3
  PCs via `Vh[i] @ W_U.T`.
- Plot a `(layers x experts)` heatmap for `pc1_evr`.
- Print top-5 experts by SOMP EVR with their first 3 PC token lists.

Feasibility: easy. Use randomized or truncated SVD only if full SVD is slow.

### 1c. Redundancy Without New Runs

Start with:
- SOMP top-token overlap, but use it as a weak signal only.
- PC1 cosine similarity within each layer.
- Mean-output cosine similarity.

Then rank candidate pairs by agreement across metrics. Avoid claiming true
redundancy until ablation confirms it.

## 2. Routing Count Pass

Add `python main.py route --n_docs N --content_tokens`.

Capture selected expert indices for all real, non-padding tokens and save:
- `counts.npy`: `(layers, experts)`
- `coactivation.npy`: `(layers, experts, experts)` for same-token expert overlap
- optional `router_entropy.npy`: mean router entropy per layer/expert if logits are exposed

Why this matters: OLMoE routing is token-conditional and prior work reports strong
routing specialization. Full-token counts let us separate "expert writes a semantic
direction" from "router actually sends that semantic content to the expert."

Implementation notes:
- Prefer selected expert indices for the first version; raw router logits can come later.
- Add `get_router_logits(layer)` only after verifying nnsight node names per adapter.
- Do not store per-token logits by default.

Feasibility: medium. It requires another forward pass, but storage is small.

## 3. Causal Validation

### 3a. Expert Ablation

Add `python main.py ablate --experts L:E,... --dataset ...`.

Patch selected expert outputs to zero and measure:
- next-token loss/perplexity on a small clean set
- task-specific logprob deltas for a targeted concept
- matched random controls with the same layer/count distribution

Use this to validate redundancy candidates from section 1c. Do not rank
dispensability using EVR alone; combine activation frequency, redundancy score,
and random-control ablation effect.

Feasibility: medium-high. Requires careful nnsight patching, but no training.

### 3b. Concept Steering

Start reversible, then consider weight edits.

Inference-time patch:
```python
v = F.normalize(concept_direction, dim=0)
h = h - alpha * (h @ v)[:, None] * v[None, :]
```

Use concept directions from either:
- restricted-dictionary SOMP atoms
- mean of a curated concept word list in `W_U`
- PC directions if they align with the concept

Evaluate first with logprob probes and clean controls. For toxicity, use a small,
fixed prompt list and report token-level deltas before doing generation metrics.

Permanent weight edit:
- Only edit after the reversible patch works.
- For OLMoE `down_proj.weight` is expected to be `[d_model, d_ff]`; project the
  output rows with `W <- W - v[:, None] @ (v[None, :] @ W)`.
- Assert shape and orientation in code; adapters may differ.

Feasibility: medium. Scientific risk is high: one toxic token direction is not a
toxicity concept.

## 4. Routing-Sensitive Attention Heads

Question: which attention heads causally shift downstream expert routing?

Prototype on 1-5 clean/corrupted prompt pairs:
- patch attention head output from corrupted into clean
- measure downstream routing change by selected-expert Jaccard first
- add router-logit KL only after raw logits are exposed reliably
- also measure output logit-difference for the target concept

Expected output: `(attention_layer x head)` heatmap of routing influence.

Feasibility: medium-low. Interesting, but do it after sections 1-3 because it needs
new adapter methods for head outputs and router observations.

## 5. Additional Interesting Directions

- Standing-committee audit: find experts or expert groups that dominate across
  domains instead of specializing. This directly tests whether Expert Pursuit is
  finding semantic experts or just frequent routing committees.
- Misrouting/counterfactual routes: for selected tokens, compare the standard top-8
  route to sampled equal-compute alternative expert sets. This is heavier but now
  directly relevant to recent MoE routing work.
- Content-token averaging: implement after the route pass. It should average expert
  outputs over user/content tokens with zero contribution where the expert was not
  selected.
- Patchscope/logit-lens validation for top findings only, as noted in `TODO.md`.

## Suggested Order

1. Mean projection + spectrum + plots.
2. Redundancy ranking from existing HDF5.
3. Routing count pass.
4. Expert ablation with random controls.
5. Reversible concept steering.
6. Routing-sensitive attention-head circuits.

## Sources Checked

- Head Pursuit paper/code: sparse decomposition over unembedding directions and
  head-level interventions.
  https://openreview.net/pdf/4f1766fc48f5e8623e50a69bf4f365a90dc2420c.pdf
  https://github.com/lorenzobasile/HeadPursuit
- OLMoE paper/model card/config: 16 layers, 64 experts, top-8 routing, 2048 hidden
  size, 50304 vocab, 1024 expert hidden size.
  https://arxiv.org/abs/2409.02060
  https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct
- Recent routing work worth tracking:
  https://arxiv.org/abs/2605.07260
  https://arxiv.org/abs/2601.03425
