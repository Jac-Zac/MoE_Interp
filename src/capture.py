from typing import Any, Dict, List

import nnsight


def capture_expert_activations(model, prompts):

    with model.trace() as tracer:
        hidden_dims = []
        for prompt_idx, prompt in enumerate(prompts):
            with tracer.invoke(prompt):
                # Get expert indices - shape is typically [batch, seq_len, top_k]
                # expert_indices = layer.mlp.experts.source.expert_idx.save()
                hidden_dims.append(model.model.layers[0].mlp.gate.output)

        nnsight.save(hidden_dims)

    return hidden_dims
