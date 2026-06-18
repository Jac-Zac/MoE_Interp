"""Causal expert-circuit experiments (require a model forward pass via nnsight).

The toxic-circuit study asks whether the experts Expert Pursuit flags as toxicity
specialists are also *causally* responsible for toxic continuations, at three
granularities:

- ``toxicity``  — whole-expert gate ablation (single + group-vs-control); the coarse view.
- ``direction`` — a diff-of-means toxic *direction*: steering, project-out, logit-lens
  and SOMP read-outs.
- ``attribution`` — gradient AtP/RelP over router gates (one backward pass) + faithfulness.
- ``neuron``    — gradient-free per-neuron contributions in the privileged neuron basis.

``pipeline.run_toxic_circuit`` runs all three on a loaded model and returns a dict (used by
``main.py circuit``); ``notebooks/notebook_circuits.py`` drives the same functions
cell-by-cell for interactive inspection.
"""

from moe_interp.circuit.attribution import faithfulness, gate_attribution, top_experts
from moe_interp.circuit.direction import (
    collect_last_token_residuals,
    project_out,
    read_direction,
    steer_sweep,
)
from moe_interp.circuit.neuron import (
    name_neurons,
    neuron_direction_attribution,
    sparsity,
    top_neurons,
)
from moe_interp.circuit.pipeline import default_prompts, run_toxic_circuit
from moe_interp.circuit.toxicity import (
    ExpertAblationResult,
    SetAblationResult,
    build_toxic_token_ids,
    run_expert_ablation,
    run_set_ablation,
    toxic_logit_score,
    toxic_probability,
)

__all__ = [
    # toxicity (whole-expert ablation)
    "ExpertAblationResult",
    "SetAblationResult",
    "build_toxic_token_ids",
    "run_expert_ablation",
    "run_set_ablation",
    "toxic_logit_score",
    "toxic_probability",
    # direction (Method A)
    "collect_last_token_residuals",
    "steer_sweep",
    "project_out",
    "read_direction",
    # gate attribution (Method B)
    "gate_attribution",
    "top_experts",
    "faithfulness",
    # neuron basis (Method B')
    "neuron_direction_attribution",
    "top_neurons",
    "name_neurons",
    "sparsity",
    # orchestration
    "run_toxic_circuit",
    "default_prompts",
]
