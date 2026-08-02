# Harmful-content proxy circuit

This exploratory pipeline compares three expert selectors:

- **SOMP:** association between pursuit atoms and the offensive-word lexicon.
- **Gate-AtP:** first-order `gate × gradient` estimate of each expert's direct effect.
- **Random:** a layer-matched control.

The metric is an offensive-token logit proxy, not an independent toxicity classifier.

## Run

```bash
python main.py pursuit --concept offensive
python notebooks/circuits/localize.py
python scripts/cineca/downweight_runner.py
```

The localization step compares gate-AtP with exhaustive per-slot gate ablation. The intervention
step scales selected post-top-k router weights during greedy generation; it does not renormalize
the remaining weights or select replacement experts.

Results are written below `data/<model>/circuit/`:

- `attribution/` — gate-AtP grids and heatmaps
- `compare/` — gate-AtP versus exhaustive gate-ablation results
- `downweight/` — per-prompt scores and bootstrap confidence intervals

The current runner supports only the offensive-word protocol on a RealToxicityPrompts split.
Other concepts require their own committed prompt source.

See the report for the complete methodology, results, and limitations.
