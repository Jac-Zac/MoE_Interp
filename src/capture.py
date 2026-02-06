"""Expert activation extraction utilities."""

from pathlib import Path
from typing import Optional

import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import DocumentTrace
from src.checkpoint import save_document


def capture_moe_activations(
    model: LanguageModel,
    docs: list[list[int]],
    doc_ids: list[int],
    store_freq: int = 10,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Capture MoE activations for documents.

    Documents are processed individually and saved to disk every store_freq
    documents. Each document is stored independently with its natural sequence
    length (no padding).

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

    saved_files = []
    traces_buffer = []

    for doc_idx, (doc, doc_id) in enumerate(
        tqdm(zip(docs, doc_ids), total=len(docs), desc="Processing docs")
    ):
        # Process document individually to avoid position embedding issues
        with torch.no_grad(), model.trace([doc]) as tracer:
            layer_indices, layer_weights = [], []

            for layer in model.model.layers:
                # Get routing info from the gate
                # top_k_weights: weight for each expert
                # top_k_indices: expert id active for each token
                # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
                _, weights, indices = layer.mlp.source.self_gate_0.output
                layer_indices.append(indices)
                layer_weights.append(weights)

            # Stack: [n_layers, seq_len, k]
            indices = torch.stack(layer_indices, dim=0)
            weights = torch.stack(layer_weights, dim=0)

            nnsight.save(indices)
            nnsight.save(weights)

        trace = DocumentTrace(
            expert_indices=indices,
            expert_weights=weights,
            doc_id=doc_id,
        )
        traces_buffer.append(trace)

        # Save to disk every store_freq documents
        if len(traces_buffer) >= store_freq:
            for t in traces_buffer:
                filepath = save_document(t, output_dir)
                saved_files.append(filepath)
            traces_buffer = []
            tqdm.write(
                f"Saved {store_freq} documents to disk (total: {len(saved_files)})"
            )

    # Save any remaining documents
    if traces_buffer:
        for t in traces_buffer:
            filepath = save_document(t, output_dir)
            saved_files.append(filepath)
        tqdm.write(
            f"Saved final {len(traces_buffer)} documents to disk (total: {len(saved_files)})"
        )

    return saved_files
