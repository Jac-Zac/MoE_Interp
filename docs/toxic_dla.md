# Toxic-expert Direct Logit Attribution (no model)

Which experts **write toward toxic vocabulary**, computed *only* from the stored expert
activations and the unembedding — no model forward, no gradients. Runs in seconds on the
existing extractions.

```bash
python main.py toxic-dla [--dataset pile10k] [--min_activations N] [--max_rows N]
```

Writes `data/<model>/circuit/dla/<dataset>/`: `dla_grid.npy`, `dla_grid.html` (layer×expert
heatmap, red = writes toward toxic tokens), `dla_top_experts.json`.

## Method (`src/moe_interp/analysis/toxic_dla.py`)

Following *The Expert Strikes Back* (arXiv:2604.02178), each expert's toxic score is its
stored output contribution projected onto the toxic-logit direction, averaged over its
tokens:

```
score(l, e) = mean_tokens( contribution · toxic_dir )
toxic_dir   = mean(U[toxic_ids]) - mean(U)
```

The stored activation is already `expert_forward × gate × RMSNorm` (see `capture.py`), so
this is exactly the expert's additive push on the toxic logits. The toxic token set comes
from `CONCEPT_WORDS["offensive"]` (single-token words, ± leading space).

**Use an all-token extraction** (`pile10k`): `rtp`/`triviaqa` are last-token captures with
only a handful of rows per expert, too sparse to score.

## Relation to causal patching
This is the cheap, activations-only **proxy**; the causal ground truth is gate-ablation
patching (`circuit/patching.py`, on the `feat/expert-circuit` branch). DLA measures *writing
toward* toxic tokens; ablation measures *causal effect of removal* — they agree on the
strongest late-layer experts but ablation additionally reveals suppressor experts (negative
effect) that DLA cannot see.
