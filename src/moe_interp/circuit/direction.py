"""Method A — a distributed toxicity *direction* (not a single expert).

Whole-expert ablation is the coarsest possible basis: it removes a ~200-effective-rank
chunk of capacity to delete what is plausibly a ~1-D feature, so it mostly inflicts
generic damage (see the null in ``toxicity.py``). Following Transluce's "circuits are
sparse in the neuron basis", we instead isolate a *direction* in the residual stream:

    v = mean(h | toxic-eliciting prompts) - mean(h | neutral prompts)      (diff-of-means)

and test it causally three ways:
  - **steering**: add alpha*v at a layer for all positions, sweep alpha -> the toxic-token
    logit score should move monotonically;
  - **project-out**: remove the v component (h <- h - (h.v_hat) v_hat); on toxic prompts
    the toxic score should drop, while on neutral prompts the next-token distribution
    should barely move (low KL) -- the discriminating test whole-expert ablation fails;
  - **read-out**: project v onto the unembedding to see which tokens it writes, and rank
    experts by how much their stored mean output aligns with v (the distributed set).

All interventions are inside one nnsight trace; layers are touched in forward order.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_interp.circuit.toxicity import right_padded, toxic_logit_score


def collect_last_token_residuals(
    model, prompts: list[list[int]], layer: int, batch_size: int = 8
) -> torch.Tensor:
    """Residual stream at ``layer`` output, gathered at each prompt's last real token."""
    chunks: list[torch.Tensor] = []
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = torch.tensor([len(t) for t in batch])
            with torch.no_grad(), model.trace(batch):
                # decoder layer .output is the bare hidden-states tensor (B, T, D)
                hs = model.model.layers[layer].output.save()
            rows = torch.arange(hs.shape[0])
            chunks.append(hs[rows, lengths - 1].float().cpu())
    return torch.cat(chunks)


def _last_logits(model, prompts, layer, edit, batch_size):
    """Run prompts, apply ``edit(hidden_states)`` in place at ``layer``, return last-tok logits."""
    chunks = []
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = torch.tensor([len(t) for t in batch])
            with torch.no_grad(), model.trace(batch):
                if edit is not None:
                    h = model.model.layers[layer].output
                    h[:] = edit(h)
                logits = model.output.logits.save()
            rows = torch.arange(logits.shape[0])
            chunks.append(logits[rows, lengths - 1].float().cpu())
    return torch.cat(chunks)


def steer_sweep(
    model,
    toxic_prompts: list[list[int]],
    layer: int,
    v: torch.Tensor,
    alphas: list[float],
    toxic_ids: list[int],
    batch_size: int = 8,
) -> list[float]:
    """Mean toxic-logit score on toxic prompts after adding ``alpha * v`` at ``layer``."""
    out: list[float] = []
    for a in alphas:
        if a == 0.0:
            edit = None
        else:

            def edit(h, a=a):
                return h + a * v.to(h.device, h.dtype)

        logits = _last_logits(model, toxic_prompts, layer, edit, batch_size)
        out.append(float(toxic_logit_score(logits, toxic_ids).mean()))
    return out


def project_out(
    model,
    toxic_prompts: list[list[int]],
    neutral_prompts: list[list[int]],
    layer: int,
    v: torch.Tensor,
    toxic_ids: list[int],
    batch_size: int = 8,
) -> tuple[float, float]:
    """Remove the ``v`` component; return (toxic-score drop, neutral next-token KL)."""

    def edit(h, v=v):
        vh = F.normalize(v.to(h.device, h.dtype), dim=0)
        return h - (h @ vh).unsqueeze(-1) * vh

    base_tox = toxic_logit_score(
        _last_logits(model, toxic_prompts, layer, None, batch_size), toxic_ids
    )
    proj_tox = toxic_logit_score(
        _last_logits(model, toxic_prompts, layer, edit, batch_size), toxic_ids
    )
    toxic_delta = float((base_tox - proj_tox).mean())

    base_n = _last_logits(model, neutral_prompts, layer, None, batch_size)
    proj_n = _last_logits(model, neutral_prompts, layer, edit, batch_size)
    kl = F.kl_div(
        F.log_softmax(proj_n, dim=-1), F.softmax(base_n, dim=-1), reduction="batchmean"
    )
    return toxic_delta, float(kl)


def read_direction(v: torch.Tensor, dictionary: torch.Tensor, tokenizer, k: int = 12):
    """Top vocabulary tokens the direction ``v`` writes (logit-lens of the direction)."""
    scores = dictionary @ F.normalize(v.float(), dim=0)
    idx = torch.topk(scores, k).indices.tolist()
    return [tokenizer.decode([i]) for i in idx]
