# SPDX-License-Identifier: Apache-2.0
"""Differentiable shared key/value attention with causal suffix history."""
import torch


def attention(queries, memory, history, positions):
    cache = torch.cat([memory, history], dim=0)
    scores = torch.einsum('rhd,nd->rhn', queries, cache) * 0.0625
    row_ids = torch.arange(history.shape[0], device=queries.device)
    mask = row_ids[None, :] <= positions[:, None]
    full = torch.cat([torch.ones((queries.shape[0], memory.shape[0]), device=queries.device, dtype=torch.bool), mask], dim=1)
    scores = scores.masked_fill(~full[:, None, :], -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum('rhn,nd->rhd', weights, cache)
