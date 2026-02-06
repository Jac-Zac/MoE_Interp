"""Simple document-based storage for MoE traces."""

from pathlib import Path
from typing import Optional

from safetensors.torch import load_file, save_file

from src.cache import DocumentTrace


def get_data_dir() -> Path:
    """Get data directory from MOE_DATA_DIR, DATA_DIR, or ./data."""
    import os

    data_dir = os.environ.get("MOE_DATA_DIR") or os.environ.get("DATA_DIR", "./data")
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_document(trace: DocumentTrace, output_dir: Path) -> Path:
    """Save a single document trace to disk.

    Args:
        trace: DocumentTrace to save
        output_dir: Directory to save the file

    Returns:
        Path to saved file
    """
    filename = f"doc_{trace.doc_id:06d}.safetensors"
    filepath = output_dir / filename

    tensors = {
        "expert_indices": trace.expert_indices,
        "expert_weights": trace.expert_weights,
    }

    save_file(tensors, filepath)
    return filepath


def load_document(doc_id: int, data_dir: Optional[Path] = None) -> DocumentTrace:
    """Load a document trace by its ID.

    Args:
        doc_id: Document ID to load
        data_dir: Directory containing trace files

    Returns:
        DocumentTrace for the requested document

    Raises:
        FileNotFoundError: If document not found
    """
    if data_dir is None:
        data_dir = get_data_dir()

    filepath = data_dir / f"doc_{doc_id:06d}.safetensors"

    if not filepath.exists():
        raise FileNotFoundError(f"Document {doc_id} not found at {filepath}")

    tensors = load_file(filepath)

    return DocumentTrace(
        expert_indices=tensors["expert_indices"],
        expert_weights=tensors["expert_weights"],
        doc_id=doc_id,
    )


def list_documents(data_dir: Optional[Path] = None) -> list[int]:
    """List all available document IDs."""
    if data_dir is None:
        data_dir = get_data_dir()

    doc_ids = []
    for filepath in sorted(data_dir.glob("doc_*.safetensors")):
        # Extract doc_id from filename (doc_{id:06d}.safetensors)
        try:
            doc_id = int(filepath.stem.split("_")[1])
            doc_ids.append(doc_id)
        except (ValueError, IndexError):
            continue

    return doc_ids


def load_all_documents(data_dir: Optional[Path] = None) -> dict[int, DocumentTrace]:
    """Load all documents from a directory.

    Args:
        data_dir: Directory containing trace files

    Returns:
        Dictionary mapping doc_id to DocumentTrace
    """
    if data_dir is None:
        data_dir = get_data_dir()

    documents = {}
    for doc_id in list_documents(data_dir):
        try:
            trace = load_document(doc_id, data_dir)
            documents[doc_id] = trace
        except FileNotFoundError:
            continue

    return documents
