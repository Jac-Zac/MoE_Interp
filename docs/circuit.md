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
