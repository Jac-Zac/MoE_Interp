# Causal toxic-expert circuit

Loads the model (via nnsight) and measures **which experts causally drive toxic
continuations**, by intervening on the router gates. Complements the gradient-free
`analysis/toxic_dla.py` (which only reads stored activations).

```bash
python main.py circuit [--model M] [--layers L ...] [--batch_size B] [--n_prompts N]
```

Default behaviour: builds the toxic-eliciting seed prompts, then sweeps every routed
`(layer, expert)`, zeroing its gate (an ablation patch) and recording the change in the
toxic-logit metric. Writes to `data/<model>/circuit/patching/`:
`patching_grid.npy`, `patching_grid.html` (layer×expert heatmap, red = expert promotes
toxicity), `top_experts.json`.

## Faithfulness comparison (vs the causal patching grid)

```bash
python main.py circuit            # build the causal ground-truth grid first
python main.py circuit-compare    # score the cheap attributors against it
```

Pooled Pearson r of each method's per-expert effect against the 16-layer patching grid
(913 scored experts, OLMoE pile10k toxic prompts):

| method | cost | r vs patching |
|---|---|---|
| **gate-AtP** (`attribution.py`) | 1 backward pass | **+0.80** (per-layer up to +0.98) |
| RelP, neuron basis (`relp.py`) | per-layer forward | +0.07 |
| DLA, diff-of-means (neuron) | per-layer forward | +0.09 |
| DLA, activations only (`analysis/toxic_dla.py`) | no model | +0.005 |

Takeaways: **gradient attribution patching over the router gates is both cheap and faithful**
— one backward pass recovers the expensive causal grid. The direction-based RelP/DLA methods
only track the causal effect at the final write-to-vocab layer (L14), because they measure an
expert's *direct* push on the toxic logits and miss its downstream causal paths (which the
gate gradient captures). This is the opposite of the RelP paper's AtP≪RelP result, for two
reasons: OLMoE's router gate is a clean differentiable leaf (so AtP is not noisy here, unlike
their LayerNorm-bottlenecked MLP nodes), and the RelP here is a deliberately simple
approximation (single skip-dominant relevance direction + last-token direct effect; a full
multi-layer LRP backward is future work). NB: AtP at the *neuron* basis is unavailable —
nnsight 0.7 does not provide `dL/d(residual)` through the traced forward (`MissedProviderError`),
the same limitation that makes `neuron.py` gradient-free.

## Methods in `src/moe_interp/circuit/`

- `patching.py` — the brute-force **causal grid** (one forward per routed expert; experts
  never routed in the batch are skipped). The ground truth.
- `attribution.py` — **AtP** gradient attribution patching over the gates: estimates the
  whole grid from one backward pass (`gate · dL/dgate`). `faithfulness()` is its Pearson r
  vs the true ablation.
- `toxicity.py` — single-expert and set ablation, toxic prompts, the toxic-logit metric.
- `direction.py`, `neuron.py`, `pipeline.py` — diff-of-means steering, neuron-basis
  attribution, and the combined `run_toxic_circuit` study.

## Running locally vs on Orfeo

OLMoE-1B-7B loads on Apple MPS in ~30 s (~13 GB weights; needs ~16 GB free RAM) and a
single ablation forward is ~2 s, so the **full 16-layer grid (~15-20 min) runs on a Mac**.
If RAM is tight, restrict with `--layers` / `--n_prompts`, or run on the GPU cluster:

```bash
# On Orfeo (CUDA): same command, the model pins to cuda via get_device()
DATA_DIR=$SCRATCH/data python main.py circuit --batch_size 16
# then pull data/<model>/circuit/ back to inspect the heatmap locally
```

The intervention point is the fused-experts boundary `layer.mlp.experts.inputs[0]` →
`(hidden_states, top_k_index, top_k_weights)`; the gate weights are the only per-expert
differentiable/interventionable node exposed on transformers ≥ 5.9.
