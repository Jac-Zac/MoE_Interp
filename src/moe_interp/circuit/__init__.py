"""Causal toxic-expert circuit (requires a model forward pass via nnsight).

The question: which experts are *causally* responsible for toxic continuations — and can
we suppress toxicity by acting on them? Modules:

- ``prompts``     — toxic-eliciting and matched neutral seed prompts.
- ``toxicity``    — toxic-logit metric, shared ablation plumbing, whole-set significance test.
- ``patching``    — the per-(layer, expert) causal effect grid (one forward per expert).
- ``attribution`` — gate-AtP: gradient estimate of the whole grid in one backward pass.
- ``compare``     — faithfulness of cheap attributors vs the causal patching grid.
- ``intervene``   — generation-time knockout / project-out to suppress toxicity.
- ``steer``       — orchestrates the intervention experiment (incl. the diff-of-means
  toxic direction from last-token residuals).
- ``report``      — assembles the artifacts into one self-contained HTML report.

The gradient-free expert *classifier* counterpart (no model) is
``moe_interp.analysis.toxic_dla``.
"""
