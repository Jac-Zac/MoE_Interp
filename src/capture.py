"""Expert Pursuit activation extraction.

Captures gated expert outputs via nnsight tracing, aggregates to
per-document means (averaging only question-content tokens), and
streams to HDF5 via ExpertActivationStore.
"""

from pathlib import Path

import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import ExpertActivationStore
from src.data import TokenizedQuestion


def _aggregate_document(
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    active_experts_per_layer: list[torch.Tensor],
    token_indices_per_layer: list[list[torch.Tensor]],
    raw_outputs_per_layer: list[list[torch.Tensor]],
    top_k_pos_per_layer: list[list[torch.Tensor]],
    n_experts: int,
    d_model: int,
    content_start: int = 0,
    content_end: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-expert mean gated outputs for one document.

    Only tokens within [content_start, content_end) are included in the
    average, matching HeadPursuit's approach of excluding chat template
    markers from aggregation.

    Args:
        expert_indices: [n_layers, seq_len, k] routing indices
        expert_weights: [n_layers, seq_len, k] gating weights
        active_experts_per_layer: list of active expert ID tensors per layer
        token_indices_per_layer: per-layer list of token index tensors
        raw_outputs_per_layer: per-layer list of raw down_proj output tensors
        top_k_pos_per_layer: per-layer list of top-k position tensors
        n_experts: total number of experts
        d_model: hidden dimension
        content_start: first question-content token index
        content_end: one past last question-content token index (None = all)

    Returns:
        expert_means: [n_layers, n_experts, d_model]
        routing_counts: [n_layers, n_experts]
    """
    n_layers = expert_indices.shape[0]
    if content_end is None:
        content_end = expert_indices.shape[1]

    expert_means = torch.zeros(n_layers, n_experts, d_model)
    routing_counts = torch.zeros(n_layers, n_experts, dtype=torch.long)

    for layer_idx in range(n_layers):
        active_experts = active_experts_per_layer[layer_idx]
        token_indices_list = token_indices_per_layer[layer_idx]
        raw_outputs_list = raw_outputs_per_layer[layer_idx]
        top_k_pos_list = top_k_pos_per_layer[layer_idx]

        for i, expert_id_tensor in enumerate(active_experts):
            expert_id = int(expert_id_tensor.item())
            token_idxs = token_indices_list[i]
            raw_output = raw_outputs_list[i]
            k_positions = top_k_pos_list[i]

            if token_idxs.numel() == 0:
                continue

            # Filter to content tokens only (exclude chat template markers)
            content_mask = (token_idxs >= content_start) & (token_idxs < content_end)
            if not content_mask.any():
                continue

            token_idxs = token_idxs[content_mask]
            raw_output = raw_output[content_mask]
            k_positions = k_positions[content_mask]

            # Gated output: g_e(x) * f_e(x)
            gate_weights = expert_weights[layer_idx, token_idxs, k_positions]
            gated = gate_weights.unsqueeze(-1) * raw_output  # [n_tokens, d_model]

            expert_means[layer_idx, expert_id] = gated.mean(dim=0)
            routing_counts[layer_idx, expert_id] = token_idxs.numel()

    return expert_means, routing_counts


def capture_document(
    model: LanguageModel,
    question: TokenizedQuestion | list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture MoE activations for a single document via nnsight.

    Args:
        model: nnsight LanguageModel instance
        question: TokenizedQuestion (with content boundaries) or raw token IDs.
            When raw token IDs are passed, all tokens are averaged.

    Returns:
        expert_means: [n_layers, n_experts, d_model]
        routing_counts: [n_layers, n_experts]
        expert_weights_all: [n_layers, seq_len, k]
    """
    if isinstance(question, TokenizedQuestion):
        doc = question.token_ids
        content_start = question.content_start
        content_end = question.content_end
    else:
        doc = question
        content_start = 0
        content_end = None

    layer_indices: list = []
    layer_weights: list = []
    active_experts_per_layer: list[torch.Tensor] = []
    token_indices_per_layer: list[list[torch.Tensor]] = []
    raw_outputs_per_layer: list[list[torch.Tensor]] = []
    top_k_pos_per_layer: list[list[torch.Tensor]] = []

    with torch.no_grad(), model.trace([doc]) as tracer:
        for layer in model.model.layers:
            # Get routing info from the gate
            _, weights, indices = layer.mlp.source.self_gate_0.output
            layer_indices.append(indices)
            layer_weights.append(weights)

            # Get active experts for this layer
            expert_hit = layer.mlp.experts.source.nonzero_0.output
            num_experts_total = model.config.num_experts
            active_experts = expert_hit[expert_hit != num_experts_total].squeeze(-1)
            num_iters = active_experts.numel()

            # Capture per-expert data
            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output
                down_proj = layer.mlp.experts.source.nn_functional_linear_1.output
                token_indices_list.append(token_idx)
                down_projs_list.append(down_proj)
                top_k_pos_list.append(top_k_pos)

            active_experts_per_layer.append(active_experts)
            token_indices_per_layer.append(token_indices_list)
            raw_outputs_per_layer.append(down_projs_list)
            top_k_pos_per_layer.append(top_k_pos_list)

        indices_t = torch.stack(layer_indices, dim=0)
        weights_t = torch.stack(layer_weights, dim=0)

        nnsight.save(indices_t)
        nnsight.save(weights_t)
        nnsight.save(active_experts_per_layer)
        nnsight.save(token_indices_per_layer)
        nnsight.save(raw_outputs_per_layer)
        nnsight.save(top_k_pos_per_layer)

    n_experts = model.config.num_experts
    d_model = model.config.hidden_size

    expert_means, routing_counts = _aggregate_document(
        expert_indices=indices_t,
        expert_weights=weights_t,
        active_experts_per_layer=active_experts_per_layer,
        token_indices_per_layer=token_indices_per_layer,
        raw_outputs_per_layer=raw_outputs_per_layer,
        top_k_pos_per_layer=top_k_pos_per_layer,
        n_experts=n_experts,
        d_model=d_model,
        content_start=content_start,
        content_end=content_end,
    )

    return expert_means, routing_counts, weights_t


def encode_dataset(
    model: LanguageModel,
    questions: list[TokenizedQuestion],
    output_dir: Path,
) -> Path:
    """Encode a dataset: capture gated expert outputs and save to HDF5.

    Args:
        model: nnsight LanguageModel
        questions: List of TokenizedQuestion from load_triviaqa()
        output_dir: Root directory for HDF5 output

    Returns:
        Path to output directory
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    d_model = model.config.hidden_size

    output_dir = Path(output_dir)

    with ExpertActivationStore(
        root_dir=output_dir,
        n_layers=n_layers,
        n_experts=n_experts,
        d_model=d_model,
        n_docs_estimate=len(questions),
    ) as store:
        for i, question in enumerate(tqdm(questions, desc="Encoding")):
            expert_means, routing_counts, _ = capture_document(model, question)
            store.add_document(expert_means, routing_counts, question.source_idx)

            if (i + 1) % 100 == 0:
                store.flush()
                tqdm.write(f"Flushed {i + 1}/{len(questions)} documents to disk")

    return output_dir
