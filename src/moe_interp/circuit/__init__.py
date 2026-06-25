"""Causal toxic-expert circuit (requires a model forward pass via nnsight).

The question: which experts are *causally* responsible for toxic continuations — and can
we suppress toxicity by acting on them? Modules:

- ``prompts``     — RealToxicityPrompts split: ``rtp_split`` gives disjoint train (identify)
  / test (evaluate) eliciting+neutral sets so the intervention is scored out-of-sample.
- ``toxicity``    — toxic-logit metric, shared ablation plumbing, whole-set significance test.
- ``patching``    — the per-(layer, expert) causal effect grid (one forward per expert).
- ``attribution`` — gate-AtP: gradient estimate of the whole grid in one backward pass.
- ``compare``     — faithfulness of cheap attributors vs the causal patching grid.
- ``intervene``   — generation-time knockout / project-out to suppress toxicity.
- ``steer``       — orchestrates the intervention experiment (incl. the diff-of-means
  toxic direction from last-token residuals).
- ``report``      — assembles the artifacts into one self-contained HTML report.

The correlational expert *classifier* counterpart (no model) is SOMP / Expert Pursuit
(``moe_interp.pursuit``), scored on the offensive concept word list.
"""
