"""Scaled dot-product attention: a reference implementation and a tiled online-softmax variant.

Both take q [B, H, Tq, D], k/v [B, H, Tk, D] and `q_offset`, the absolute position of the
first query. Key j sits at absolute position j, so with a KV cache (Tq < Tk) a query at
position q_offset + i may attend to keys 0 .. q_offset + i. This one rule covers prefill
(q_offset = 0, Tq = Tk), single-token decoding and multi-token speculative verification.
"""
import math

import torch


def causal_mask(tq, tk, q_offset, device):
    """Boolean [Tq, Tk] mask, True where attention is allowed."""
    q_pos = torch.arange(tq, device=device)[:, None] + q_offset
    k_pos = torch.arange(tk, device=device)[None, :]
    return k_pos <= q_pos


def reference_attention(q, k, v, q_offset=0, causal=True):
    """softmax(QK^T / sqrt(d) + mask) V with max subtraction, computed in fp32."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        scores = scores.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device), float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)  # numerical stability
    p = scores.exp()
    p = p / p.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v.float()).to(q.dtype)


def tiled_attention(q, k, v, q_offset=0, causal=True, block_q=64, block_k=64):
    """FlashAttention-style tiling: never materialises the full [Tq, Tk] score matrix.

    For each query block we stream over key/value blocks and keep, per query row,
    a running max m, a running normaliser l = sum exp(s - m) and an unnormalised
    output acc = sum exp(s - m) v. When a new block raises the max to m_new, the old
    statistics are rescaled by exp(m - m_new), so the result equals the exact softmax.
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, H, Tq, D, dtype=torch.float32, device=q.device)
    kf, vf = k.float(), v.float()

    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[:, :, qs:qe].float() * scale
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, D), device=q.device)
        last_q_pos = q_offset + qe - 1
        for ks in range(0, Tk, block_k):
            if causal and ks > last_q_pos:
                break  # this and every later key block lies entirely in the future
            ke = min(ks + block_k, Tk)
            s = torch.matmul(qb, kf[:, :, ks:ke].transpose(-1, -2))
            if causal and ke - 1 > q_offset + qs:  # block straddles the diagonal
                allowed = (torch.arange(ks, ke, device=q.device)[None, :]
                           <= torch.arange(qs, qe, device=q.device)[:, None] + q_offset)
                s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # Rows whose every key so far is masked keep m = -inf; use 0 to avoid inf - inf = nan.
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            correction = torch.exp(m - m_safe)
            l = l * correction + p.sum(dim=-1, keepdim=True)
            acc = acc * correction + torch.matmul(p, vf[:, :, ks:ke])
            m = m_new
        out[:, :, qs:qe] = acc / l
    return out.to(q.dtype)


def sdpa_attention(q, k, v, q_offset=0, causal=True):
    """PyTorch's fused scaled_dot_product_attention, used as an external reference/baseline."""
    mask = causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device) if causal else None
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


ATTENTION = {"reference": reference_attention, "tiled": tiled_attention, "sdpa": sdpa_attention}
