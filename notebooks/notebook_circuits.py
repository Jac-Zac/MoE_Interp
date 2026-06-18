#!/usr/bin/env python
"""Causal toxic-circuit study on OLMoE — interactive `# %%` walkthrough.

Are the experts Expert Pursuit flags as toxicity specialists *causally* responsible for
toxic continuations? We look at three granularities, all from `moe_interp.circuit`:

  attempt 1  whole-expert gate ablation        -> null (too coarse)
  A          diff-of-means toxic direction      -> steering + project-out + read-outs
  B          gradient gate attribution (AtP)     -> ranks experts, faithfulness vs ablation
  B'         neuron-basis attribution            -> sparse distributed neuron circuit

Run as a script or step cell-by-cell:
    DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/notebook_circuits.py
"""

# %% Imports
import json

from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.analysis.common import resolve_pursuit_dir
from moe_interp.capture.cache import load_unembedding
from moe_interp.circuit import (
    build_toxic_token_ids,
    collect_last_token_residuals,
    default_prompts,
    faithfulness,
    gate_attribution,
    name_neurons,
    neuron_direction_attribution,
    project_out,
    read_direction,
    run_expert_ablation,
    run_set_ablation,
    sparsity,
    steer_sweep,
    top_experts,
    top_neurons,
)
from moe_interp.config import (
    get_default_model,
    get_device,
    get_model_dir,
    get_unembedding_dir,
    set_seed,
)

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
LAYER = 12  # residual layer for the toxic direction (toxicity SOMP specialists ~ L12-15)
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
BATCH = 6

# %% Load model + the normalized unembedding dictionary
model = LanguageModel(
    MODEL_NAME, dtype="auto", dispatch=True, device_map=str(get_device())
)
tok = model.tokenizer
dictionary = load_unembedding(get_unembedding_dir(MODEL_NAME) / "dictionary.h5").float()
toxic_ids = build_toxic_token_ids(tok)
toxic_prompts, neutral_prompts = default_prompts(tok)
print(f"Loaded {MODEL_NAME} on {get_device()} | toxic probe ids: {len(toxic_ids)}")

# %% Candidate toxicity experts from the RTP `offensive` SOMP run
pdir = resolve_pursuit_dir(MODEL_NAME, "rtp")
offensive = (pdir / "offensive" / "results.jsonl") if pdir else None
rows = [json.loads(line) for line in offensive.open()] if offensive and offensive.exists() else []
rows.sort(key=lambda r: r["evr"][-1] if r["evr"] else 0.0, reverse=True)
candidates = [(r["layer"], r["expert"]) for r in rows[:10]]
print("SOMP-flagged toxicity experts:", candidates)

# %% Attempt 1 — whole-expert ablation (expected: null, too coarse)
set_res = run_set_ablation(
    model, toxic_prompts, candidates, toxic_ids, n_controls=20, batch_size=BATCH
)
print(
    f"flagged set Δ={set_res.flagged_delta:+.4f}  control {set_res.control_mean:+.4f}"
    f"±{set_res.control_std:.4f}  z={set_res.zscore:+.2f}  beats {100 * set_res.percentile:.0f}% of controls"
)
# Persist the coarse-ablation null so scripts/build_slides.py reads it instead of a literal.
sa_path = get_model_dir(MODEL_NAME) / "analysis" / "rtp" / "set_ablation.json"
sa_path.parent.mkdir(parents=True, exist_ok=True)
sa_path.write_text(
    json.dumps(
        {
            "flagged_delta": set_res.flagged_delta,
            "control_mean": set_res.control_mean,
            "control_std": set_res.control_std,
            "zscore": set_res.zscore,
            "percentile_beats": 100 * set_res.percentile,
        },
        indent=2,
    )
)

# %% A — toxic direction (diff-of-means) + steering
res_t = collect_last_token_residuals(model, toxic_prompts, LAYER, BATCH)
res_n = collect_last_token_residuals(model, neutral_prompts, LAYER, BATCH)
v = res_t.mean(0) - res_n.mean(0)
steer = steer_sweep(model, toxic_prompts, LAYER, v, ALPHAS, toxic_ids, BATCH)
print("steering (toxic score vs alpha):", dict(zip(ALPHAS, [round(s, 3) for s in steer])))

# %% A — project-out (causal removal) + read-outs
delta, kl = project_out(model, toxic_prompts, neutral_prompts, LAYER, v, toxic_ids, BATCH)
print(f"project-out: Δtoxic={delta:+.4f} (↑ removed)  neutral KL={kl:.4f} (↓ specific)")
print("logit-lens read-out:", read_direction(v, dictionary, tok, k=12))

# %% B — gradient gate attribution (AtP/RelP) + faithfulness
attr = gate_attribution(model, toxic_prompts, toxic_ids, batch_size=BATCH)
top_e = top_experts(attr, k=12)
abl = run_expert_ablation(
    model, toxic_prompts, [(layer_i, e) for layer_i, e, _ in top_e], [], toxic_ids, BATCH
)
r = faithfulness(attr, abl, scale=len(toxic_prompts))
t = Table(title=f"Top experts by |gate attribution|   (faithfulness r={r:.2f})")
t.add_column("expert"); t.add_column("attribution")
for layer_i, expert_i, val in top_e:
    t.add_row(f"L{layer_i}E{expert_i}", f"{val:+.3f}")
print(t)

# %% B' — neuron-basis attribution + sparsity + naming
neuron_attr = neuron_direction_attribution(model, toxic_prompts, neutral_prompts, LAYER, v, BATCH)
sp = sparsity(neuron_attr)
print(
    f"sparsity: effective neurons={sp['effective_neurons']:.0f}/{sp['n_total']}  "
    f"top-20 hold {100 * sp['top20_frac']:.1f}%  top-100 hold {100 * sp['top100_frac']:.1f}% of the mass"
)
named = name_neurons(model, LAYER, top_neurons(neuron_attr, k=20), dictionary, tok, k=6)
t = Table(title=f"Top toxic neurons @ L{LAYER}  ({len({n['expert'] for n in named})} distinct experts)")
t.add_column("expert·neuron"); t.add_column("Δ"); t.add_column("write-direction tokens")
for n in named[:12]:
    t.add_row(f"E{n['expert']}·n{n['neuron']}", f"{n['attribution']:+.4f}", ", ".join(n["tokens"]))
print(t)
