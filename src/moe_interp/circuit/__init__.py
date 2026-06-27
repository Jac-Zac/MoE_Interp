"""Causal toxic-expert circuit (requires a model forward pass via nnsight).

The question: which experts are *causally* responsible for toxic continuations — and can
we suppress toxicity by acting on them? Modules:

- ``prompts``     — RealToxicityPrompts split: ``rtp_split`` gives disjoint train (identify)
  / test (evaluate) eliciting+neutral sets so the intervention is scored out-of-sample.
- ``toxicity``    — toxic-logit metric, shared ablation plumbing, whole-set significance test.
- ``attribution`` — gate-AtP: a one-backward-pass causal effect grid over (layer, expert). This
  is the causal localizer. (It was validated once against exhaustive activation patching — one
  forward per expert — which agreed closely (pooled r≈0.69, ≈0.93 late); that expensive sweep is
  no longer in the pipeline, only the cheap AtP grid is.)
- ``compare``     — the intervention propensity bar chart.
- ``intervene``   — generation-time expert knockout / expert-output steering (expert-level only).
- ``steer``       — orchestrates the intervention experiments (incl. the diff-of-means
  toxic direction from last-token residuals and per-expert output steering).
- ``report``      — assembles the artifacts into one self-contained HTML report.

The correlational expert *classifier* counterpart (no model) is SOMP / Expert Pursuit
(``moe_interp.pursuit``), scored on the offensive concept word list.
"""
