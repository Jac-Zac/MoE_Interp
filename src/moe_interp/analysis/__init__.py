"""Post-hoc analyses over captured expert activations.

These modules read the HDF5 extractions and (optionally) the SOMP pursuit results
and require no model forward pass — everything is recomputed from stored tensors:

- ``logit_lens``: bulk mean-projection logit-lens baseline + SOMP-vs-lens comparison.
"""

from moe_interp.analysis.logit_lens import run_logit_lens_comparison

__all__ = [
    "run_logit_lens_comparison",
]
