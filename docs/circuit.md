# Causal toxic-expert circuit

The full pipeline: **classify** experts by toxicity, find the **causally** responsible ones,
and **suppress** toxic generation by acting on them. Loads OLMoE via nnsight and intervenes
on the router gates. Prompts are a **RealToxicityPrompts split** (high- vs low-toxicity),
built by `circuit/prompts.py`.

This is the most experimental part of the project, so it is kept out of the `main.py` CLI:
the three stages live as `# %%` walkthroughs under `notebooks/circuits/` that drive
`moe_interp.circuit` directly (or run them end-to-end with `scripts/cineca/circuit_runner.py`).

## 1. Classify — which experts associate with toxicity (no model)

```bash
python main.py pursuit --concept offensive  # SOMP: experts whose atoms are offensive words
```

SOMP is correlational and model-free: it flags toxicity-*associated* experts (the
interpretability ranker used in the knockout comparison below).

## 2. Localize — which experts are *causally* responsible

```bash
python notebooks/circuits/localize.py   # gate-AtP causal grid (one backward pass) + heatmap
```

`localize.py` runs **gate-AtP** over the eliciting prompts: a single backward pass scores every
routed `(layer, expert)` by `gate · dL/dgate`, the first-order effect of zeroing its gate on the
toxic-logit metric → `data/<model>/circuit/attribution/atp_grid_n<N>.npy` (+ heatmap, top
experts). Positive = the expert promotes toxicity, negative = suppresses it.

> **Why not exhaustive activation patching?** Patching (zero each gate in a separate forward pass;
> one forward per expert) is the exact ground truth, but costs ~64× more. We validated gate-AtP
> against it once on the toxicity grid and the two agreed closely — pooled Pearson **r ≈ 0.69**,
> up to **≈ 0.96** in the late layers where the controllable signal lives — so the expensive
> patching sweep was dropped and the cheap AtP grid is used throughout. That frozen validation
> lives in `data/<model>/circuit/compare/faithfulness.json`. (A neuron-basis RelP variant was also
> tried and removed: it underperformed AtP — OLMoE's gate is a clean differentiable leaf — and
> nnsight 0.7 won't provide `dL/d(residual)` for a neuron-level AtP anyway.)

## 3. Intervene — suppress toxic generation

```bash
python notebooks/circuits/steer.py   # knockout / project-out vs baseline, then assemble the HTML report
```

`steer.py` ranks experts by each identification method (AtP, SOMP, random) and, during
generation, knocks out those experts or projects the toxic direction out of the residual stream,
scoring toxic-logit propensity and offensive-word rate vs baseline with a neutral-prompt
collateral check (`data/<model>/circuit/steer/`). Finding: **single-expert knockout is near-inert
for every selector — the causal (AtP) experts order correctly but top-k routing is redundant, so
knocking out a handful barely moves toxicity, and SOMP/random do ~nothing** (token association ≠
causal responsibility). Naive additive steering (a large fixed `-α·v`) tanks neutral generation,
so **project-out is the best suppressor**: it removes only the toxic direction and keeps
generation fluent. The intervention generalizes to any concept via the `CONCEPT` variable in
`steer.py`.

> All interventions act at **the router gate** (`layer.mlp.experts.inputs[0]`), the only
> per-expert node the fused kernel exposes — so they scale/zero an expert's *whole* contribution.
> Going finer (individual neurons inside the expert MLP) needs a global residual gradient the
> fused boundary won't give on nnsight 0.7, so a neuron-level AtP was tried and dropped.

## Modules in `src/moe_interp/circuit/`

- `prompts.py` — RealToxicityPrompts split (high- vs low-toxicity) eliciting/neutral prompts.
- `toxicity.py` — toxic-logit metric + shared gate-ablation plumbing.
- `attribution.py` — gate-AtP gradient attribution (`gate · dL/dgate`, one backward pass) — the causal localizer.
- `compare.py` — the intervention propensity bar chart.
- `intervene.py` — generation-time knockout / project-out / expert-output steering + scoring.
- `steer.py` — intervention orchestration + the diff-of-means toxic direction (last-token residuals).
- `report.py` — self-contained HTML report.

Driven by the `# %%` notebooks in `notebooks/circuits/` (`localize.py`, `steer.py`), or
end-to-end by `scripts/cineca/circuit_runner.py`.

## Running locally vs on Orfeo

OLMoE-1B-7B loads on Apple MPS in ~30 s (~13 GB weights; ~16 GB free RAM). gate-AtP is a single
backward pass, so the **localization grid runs on a Mac in well under a minute**; the
generation-time interventions in `steer.py` are the slower part. Tune `ATP_BATCH_SIZE` /
`STEER_BATCH_SIZE` if RAM is tight, or run on the GPU cluster (`get_device()` picks CUDA there):

```bash
DATA_DIR=$SCRATCH/data python notebooks/circuits/localize.py
# then pull data/<model>/circuit/ back to inspect locally
```

The intervention point is the fused-experts boundary `layer.mlp.experts.inputs[0]` →
`(hidden_states, top_k_index, top_k_weights)`; the gate weights are the only per-expert
differentiable/interventionable node exposed on transformers ≥ 5.9. nnsight 0.7 needs envoys
touched in forward (layer) order, and `tracer.all()` interventions during `generate` require
fixed-length output (`min_new_tokens == max_new_tokens`) or an early EOS errors.
