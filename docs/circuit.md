# Causal toxic-expert circuit

The full pipeline: **classify** experts by toxicity, find the **causally** responsible ones,
and **suppress** toxic generation by acting on them. Loads OLMoE via nnsight and intervenes
on the router gates. The gradient-free classifier counterpart is `analysis/toxic_dla.py`.

## 1. Classify — which experts associate with toxicity (no model)

```bash
python main.py toxic-dla --dataset pile10k     # DLA: writes-toward-toxic-vocab, from stored acts
python main.py pursuit  --concept offensive    # SOMP: experts whose atoms are offensive words
```

Both are correlational and model-free. DLA writes `data/<model>/circuit/dla/<dataset>/`.

## 2. Localize — which experts are *causally* responsible

```bash
python main.py circuit          # causal ground truth: ablate every expert, one forward each
python main.py circuit-compare  # is the cheap gradient method faithful to it?
```

`circuit` sweeps every routed `(layer, expert)`, zeros its gate, and records the change in the
toxic-logit metric → `data/<model>/circuit/patching/` (`patching_grid.npy`, heatmap, top
experts). `circuit-compare` scores cheap attributors against that grid (pooled Pearson r over
913 scored experts, pile10k):

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
python main.py circuit-steer    # knockout / down-weight / steer during generation, vs baseline
python main.py circuit-report   # assemble everything into one self-contained HTML report
```

`circuit-steer` ranks experts by each identification method (AtP, SOMP, DLA, patching, random)
and, during generation, knocks out / down-weights / steers, scoring toxic-logit propensity and
offensive-word rate vs baseline with a neutral-prompt collateral check
(`data/<model>/circuit/steer/`). Finding: **AtP-knockout reduces toxic propensity ~19% with
minimal collateral; patching-knockout also works; SOMP/DLA/random knockout do ~nothing**
(token association ≠ causal responsibility); knockout is blunt (can break fluency), so
AtP-to-identify + steering-to-suppress is the better recipe (steering needs calibration — a
large fixed `-α·v` tanks neutral generation too).

## Modules in `src/moe_interp/circuit/`

- `prompts.py` — toxic / matched-neutral seed prompts.
- `toxicity.py` — toxic-logit metric, single/whole-set gate ablation.
- `patching.py` — the brute-force causal grid (one forward per routed expert).
- `attribution.py` — gate-AtP gradient attribution (`gate · dL/dgate`, one backward pass).
- `compare.py` — faithfulness (Pearson r) of cheap attributors vs the patching grid.
- `direction.py` — diff-of-means toxic direction: steering, project-out, logit-lens.
- `intervene.py` — generation-time knockout / down-weight / steer + scoring.
- `report.py` — self-contained HTML report.

## Running locally vs on Orfeo

OLMoE-1B-7B loads on Apple MPS in ~30 s (~13 GB weights; ~16 GB free RAM) and one ablation
forward is ~2 s, so the **full 16-layer grid (~15-20 min) runs on a Mac**. If RAM is tight,
restrict with `--layers` / `--n_prompts`, or run on the GPU cluster:

```bash
DATA_DIR=$SCRATCH/data python main.py circuit --batch_size 16   # CUDA via get_device()
# then pull data/<model>/circuit/ back to inspect locally
```

The intervention point is the fused-experts boundary `layer.mlp.experts.inputs[0]` →
`(hidden_states, top_k_index, top_k_weights)`; the gate weights are the only per-expert
differentiable/interventionable node exposed on transformers ≥ 5.9. nnsight 0.7 needs envoys
touched in forward (layer) order, and `tracer.all()` interventions during `generate` require
fixed-length output (`min_new_tokens == max_new_tokens`) or an early EOS errors.
