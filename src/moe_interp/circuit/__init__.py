"""Causal toxic-expert circuit (requires a model forward pass via nnsight).

The question: which experts are *causally* responsible for toxic continuations — and can
we suppress toxicity by acting on them? Modules:

- ``prompts``     — RealToxicityPrompts split: ``rtp_split`` gives disjoint train (identify)
  / test (evaluate) eliciting+neutral sets so the intervention is scored out-of-sample.
- ``concept_probe`` — concept-logit metric, shared ablation plumbing, whole-set significance test.
- ``attribution`` — gate-AtP: a one-backward-pass causal effect grid over (layer, expert); the
  causal localizer. See ``attribution.py`` for the method and its one-off patching validation.
- ``intervene``   — generation-time expert gate knockout / downweighting (expert-level only).
- ``expert_sets`` — builds the SOMP / gate-AtP / matched-random expert sets the interventions act on.
- ``downweight``  — the knockout/downweighting sweep with per-prompt bootstrap error bars.
- ``report``      — assembles the localization artifacts into one self-contained HTML report.

The correlational expert *classifier* counterpart (no model) is SOMP / Expert Pursuit
(``moe_interp.pursuit``), scored on the offensive concept word list.
"""
