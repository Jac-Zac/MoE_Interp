# Additional evaluation

These scripts extend the held-out gate-intervention study using the cached gate-AtP grid.

| Script | Purpose |
|---|---|
| `sufficiency_curve.py` | Cumulative ablation across expert-set sizes |
| `group_ablation.py` | Downweight selected experts and their frequent co-firing partners |
| `bootstrap.py` | Paired bootstrap confidence intervals |

```bash
sbatch scripts/orfeo/rigor.sh
python scripts/rigor/bootstrap.py data/*/circuit/rigor/sufficiency_offensive.json
python scripts/rigor/bootstrap.py data/*/circuit/rigor/group_ablation_offensive.json
```

These interventions scale post-top-k gate weights without renormalization or replacement routing.
They measure sensitivity to that operation; they do not by themselves identify why an effect is
small or establish router compensation.

The main missing EVR controls are a random-dictionary floor, a frequency-matched token null, and a
PCA ceiling evaluated on the same expert activation matrices.
