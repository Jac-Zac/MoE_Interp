"""Expert Pursuit activation extraction."""

from pathlib import Path
from typing import Optional

import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import DocumentTrace, ExpertTrace


def capture_document(
    model: LanguageModel,
    doc: list[int],
    doc_id: int,
) -> DocumentTrace:
    """Capture MoE activations for a single document.

    Args:
        model: nnsight LanguageModel instance
        doc: Document token IDs
        doc_id: Original document index from dataset

    Returns:
        DocumentTrace containing all routing and activation data
    """
    layer_indices: list = []
    layer_weights: list = []
    expert_traces: list[dict[int, ExpertTrace]] = []

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

            # Build ExpertTrace objects for this layer
            layer_traces: dict[int, ExpertTrace] = {}
            for i, expert_id_tensor in enumerate(active_experts):
                expert_id = int(expert_id_tensor.item())
                layer_traces[expert_id] = ExpertTrace(
                    token_indices=token_indices_list[i],
                    raw_outputs=down_projs_list[i],
                    top_k_positions=top_k_pos_list[i],
                )

            expert_traces.append(layer_traces)

        # Stack indices and weights
        indices_t = torch.stack(layer_indices, dim=0)
        weights_t = torch.stack(layer_weights, dim=0)

        nnsight.save(indices_t)
        nnsight.save(weights_t)
        nnsight.save(expert_traces)

    return DocumentTrace(
        doc_id=doc_id,
        n_layers=len(layer_indices),
        expert_indices=indices_t,
        expert_weights=weights_t,
        expert_traces=expert_traces,
    )


def capture_documents(
    model: LanguageModel,
    docs: list[list[int]],
    doc_ids: list[int],
    store_freq: int = 10,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Capture MoE activations for multiple documents.

    Args:
        model: nnsight LanguageModel
        docs: List of document token ID lists
        doc_ids: Original dataset indices for each document
        store_freq: Save to disk every N documents
        output_dir: Directory to save traces (default: ./data)

    Returns:
        List of saved filepaths
    """
    if output_dir is None:
        output_dir = Path("./data")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    traces_buffer: list[DocumentTrace] = []

    for doc, doc_id in tqdm(
        zip(docs, doc_ids), total=len(docs), desc="Processing docs"
    ):
        trace = capture_document(model, doc, doc_id)
        traces_buffer.append(trace)

        if len(traces_buffer) >= store_freq:
            for t in traces_buffer:
                saved_files.append(t.save(output_dir))
            traces_buffer = []
            tqdm.write(
                f"Saved {store_freq} documents to disk (total: {len(saved_files)})"
            )

    if traces_buffer:
        for t in traces_buffer:
            saved_files.append(t.save(output_dir))
        tqdm.write(
            f"Saved final {len(traces_buffer)} documents (total: {len(saved_files)})"
        )

    return saved_files
