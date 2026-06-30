# Causal toxic-expert circuit

The full pipeline: **classify** experts by toxicity, find the **causally** responsible ones,
and **suppress** toxic generation by acting on them. Loads OLMoE via nnsight and intervenes
on the router gates. Prompts are a **RealToxicityPrompts split** (high- vs low-toxicity),
built by `circuit/prompts.py`.

This is the most experimental part of the project, so it is kept out of the `main.py` CLI:
localization lives as a `# %%` walkthrough (`notebooks/circuits/localize.py`) and as
`scripts/cineca/circuit_runner.py`; the intervention sweep is `scripts/cineca/downweight_runner.py`.

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

## 3. Intervene — suppress toxic generation (knockout / downweighting)

```bash
python scripts/cineca/downweight_runner.py   # knockout/downweighting sweep over SOMP/AtP/random
```

`downweight_runner.py` ranks experts by each identification method (AtP, SOMP, random) at two
budgets (1% and 5% of all experts) and, during multi-token greedy generation, scales those
experts' router gate — zeroing it (**knockout**) or multiplying it by a factor (**downweighting**,
e.g. 0.9 = 10% down, 0.5 = 50% down). It scores toxic-logit propensity, offensive-word rate and a
distinct-1 coherence guard **per prompt**, so a bootstrap puts 95% CIs (error bars) on every point
and every paired delta-vs-baseline (`data/<model>/circuit/downweight/sweep_<concept>.json`).
Finding: **knockout/downweighting is near-inert for every selector — the causal (AtP) experts
order correctly but top-k routing is redundant, so removing a handful barely moves toxicity, and
SOMP/random do ~nothing** (token association ≠ causal responsibility). The sweep generalizes to any
lexical concept via `--concept`.

> All interventions act at **the router gate** (`layer.mlp.experts.inputs[0]`), the only
> per-expert node the fused kernel exposes — so they scale/zero an expert's *whole* contribution.
> Going finer (individual neurons inside the expert MLP) needs a global residual gradient the
> fused boundary won't give on nnsight 0.7, so a neuron-level AtP was tried and dropped.

## Modules in `src/moe_interp/circuit/`

- `prompts.py` — RealToxicityPrompts split (high- vs low-toxicity) eliciting/neutral prompts.
- `concept_probe.py` — toxic-logit metric + shared gate-ablation plumbing.
- `attribution.py` — gate-AtP gradient attribution (`gate · dL/dgate`, one backward pass) — the causal localizer.
- `intervene.py` — generation-time gate knockout / downweighting + scoring.
- `expert_sets.py` — builds the SOMP / gate-AtP / matched-random expert sets the interventions act on.
- `downweight.py` — the knockout/downweighting sweep with per-prompt bootstrap error bars.
- `report.py` — self-contained HTML localization report (gate-AtP heatmap + faithfulness).

Driven by the `# %%` `localize.py` notebook + `scripts/cineca/circuit_runner.py` (localization),
and `scripts/cineca/downweight_runner.py` (the intervention sweep).

## Running locally vs on Orfeo

OLMoE-1B-7B loads on Apple MPS in ~30 s (~13 GB weights; ~16 GB free RAM). gate-AtP is a single
backward pass, so the **localization grid runs on a Mac in well under a minute**; the
generation-time knockout/downweighting sweep is the slower part. Lower `--batch-size` /
`--atp-batch-size` if RAM is tight, or run on the GPU cluster (`get_device()` picks CUDA there):

```bash
DATA_DIR=$SCRATCH/data python notebooks/circuits/localize.py
# then pull data/<model>/circuit/ back to inspect locally
```

The intervention point is the fused-experts boundary `layer.mlp.experts.inputs[0]` →
`(hidden_states, top_k_index, top_k_weights)`; the gate weights are the only per-expert
differentiable/interventionable node exposed on transformers ≥ 5.9. nnsight 0.7 needs envoys
touched in forward (layer) order, and `tracer.all()` interventions during `generate` require
fixed-length output (`min_new_tokens == max_new_tokens`) or an early EOS errors.
