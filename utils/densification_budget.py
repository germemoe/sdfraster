"""Capacity and deterministic candidate selection for vertex densification."""

import torch


def split_capacity(num_vertices, budget, max_split_per_iter, target_addition):
    """Count new internal vertices allowed by both caps and the caller's limit.

    The total vertex budget includes the fixed external shell, matching train.py.
    Zero/negative remaining capacity or step caps disable growth; existing
    vertices are never removed by this helper.
    """
    return max(0, min(
        int(budget) - int(num_vertices),
        int(max_split_per_iter),
        int(target_addition),
    ))


def select_candidates(scores, candidate_mask, limit, sort_all=False):
    """Select strongest candidates; equal scores prefer smaller tet indices.

    Preserve original tet order when no truncation is needed, unless sort_all
    requests score order. Otherwise use topk to find the cutoff, choose boundary
    ties in tet order, then sort only the selected set stably.
    """
    candidates = candidate_mask.nonzero(as_tuple=True)[0]
    limit = max(0, int(limit))
    if limit == 0:
        return candidates[:0]
    if candidates.numel() <= limit and not sort_all:
        return candidates
    candidate_scores = torch.nan_to_num(
        scores[candidates], nan=-float("inf"),
        posinf=float("inf"), neginf=-float("inf"),
    )
    if candidates.numel() <= limit:
        if sort_all:
            order = torch.argsort(candidate_scores, descending=True, stable=True)
            return candidates[order]
        return candidates

    cutoff = torch.topk(candidate_scores, limit, sorted=False).values.min()
    above = (candidate_scores > cutoff).nonzero(as_tuple=True)[0]
    tied = (candidate_scores == cutoff).nonzero(as_tuple=True)[0]
    selected = torch.cat([above, tied[:limit - above.numel()]])
    order = torch.argsort(candidate_scores[selected], descending=True, stable=True)
    return candidates[selected[order]]
