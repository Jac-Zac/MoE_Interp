# Causal toxic-expert circuit

The full pipeline: **classify** experts by toxicity, find the **causally** responsible ones,
and **suppress** toxic generation by acting on them. Loads OLMoE via nnsight and intervenes
on the router gates. The gradient-free classifier counterpart is `analysis/toxic_dla.py`.

This is the most experimental part of the project, so it is kept out of the `main.py` CLI:
the three stages live as `# %%` walkthroughs under `notebooks/circuits/` that drive
`moe_interp.circuit` directly.

## 1. Classify — which experts associate with toxicity (no model)

```bash
python notebooks/circuits/dla.py            # DLA: writes-toward-toxic-vocab, from stored acts
python main.py pursuit --concept offensive  # SOMP: experts whose atoms are offensive words
```

Both are correlational and model-free. DLA writes `data/<model>/circuit/dla/<dataset>/`.

## 2. Localize — which experts are *causally* responsible

```bash
python notebooks/circuits/patching.py   # causal grid (one forward per expert) + gate-AtP faithfulness
```

`patching.py` sweeps every routed `(layer, expert)`, zeros its gate, and records the change in
the toxic-logit metric → `data/<model>/circuit/patching/` (`patching_grid.npy`, heatmap, top
experts), then scores cheap attributors against that grid (pooled Pearson r over the scored
experts):

| method | cost | r vs patching |
|---|---|---|
| **gate-AtP** (`attribution.py`) | 1 backward pass | **+0.80** (per-layer up to +0.98) |
| DLA, activations only (`analysis/toxic_dla.py`) | no model | +0.005 |

**Gate gradient attribution is cheap and faithful** — one backward pass recovers the expensive
causal grid; the correlational activation-DLA score barely correlates with causal effect.
(A neuron-basis RelP variant was tried and removed: it underperformed AtP — OLMoE's gate is a
clean differentiable leaf so AtP isn't noisy here — and nnsight 0.7 won't provide
`dL/d(residual)` for a neuron-level AtP anyway.)

## 3. Intervene — suppress toxic generation

```bash
python notebooks/circuits/steer.py   # knockout / project-out vs baseline, then assemble the HTML report
```

`steer.py` ranks experts by each identification method (AtP, SOMP, DLA, patching, random)
and, during generation, knocks out those experts or projects the toxic direction out of the
residual stream, scoring toxic-logit propensity and offensive-word rate vs baseline with a
neutral-prompt collateral check (`data/<model>/circuit/steer/`). Finding: **AtP-knockout
reduces toxic propensity with minimal collateral; patching-knockout also works; SOMP/DLA/random
knockout do ~nothing** (token association ≠ causal responsibility). Knockout is blunt (can break
fluency) and naive additive steering (a large fixed `-α·v`) tanks neutral generation, so
**project-out is the best suppressor**: it removes only the toxic direction and keeps generation
fluent. The intervention generalizes to any concept via the `CONCEPT` variable in `steer.py`.

> All interventions act at **the router gate** (`layer.mlp.experts.inputs[0]`), the only
> per-expert node the fused kernel exposes — so they scale/zero an expert's *whole* contribution.
> Going finer (individual neurons inside the expert MLP) needs a global residual gradient the
> fused boundary won't give on nnsight 0.7, so a neuron-level AtP was tried and dropped.

## Modules in `src/moe_interp/circuit/`

- `prompts.py` — toxic / matched-neutral seed prompts.
- `toxicity.py` — toxic-logit metric + shared gate-ablation plumbing.
- `patching.py` — the brute-force causal grid (one forward per routed expert).
- `attribution.py` — gate-AtP gradient attribution (`gate · dL/dgate`, one backward pass).
- `compare.py` — faithfulness (Pearson r) of cheap attributors vs the patching grid.
- `intervene.py` — generation-time knockout / project-out + scoring.
- `steer.py` — intervention orchestration + the diff-of-means toxic direction (last-token residuals).
- `report.py` — self-contained HTML report.

Driven by the `# %%` notebooks in `notebooks/circuits/` (`dla.py`, `patching.py`, `steer.py`).

## Running locally vs on Orfeo

OLMoE-1B-7B loads on Apple MPS in ~30 s (~13 GB weights; ~16 GB free RAM) and one ablation
forward is ~2 s, so the **full 16-layer grid (~15-20 min) runs on a Mac**. If RAM is tight,
restrict the sweep with the `LAYERS` / `BATCH_SIZE` knobs at the top of `patching.py`, or run
on the GPU cluster (`get_device()` picks CUDA there):

```bash
DATA_DIR=$SCRATCH/data python notebooks/circuits/patching.py
# then pull data/<model>/circuit/ back to inspect locally
```

The intervention point is the fused-experts boundary `layer.mlp.experts.inputs[0]` →
`(hidden_states, top_k_index, top_k_weights)`; the gate weights are the only per-expert
differentiable/interventionable node exposed on transformers ≥ 5.9. nnsight 0.7 needs envoys
touched in forward (layer) order, and `tracer.all()` interventions during `generate` require
fixed-length output (`min_new_tokens == max_new_tokens`) or an early EOS errors.
