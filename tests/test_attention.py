"""Tiled online-softmax attention vs the reference implementation (and PyTorch SDPA)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trialsieve.attention import reference_attention, sdpa_attention, tiled_attention  # noqa: E402

# (B, H, Tq, Tk, D, q_offset): prefill, decode step, multi-token verify, ragged blocks, Tq > block.
SHAPES = [(1, 1, 1, 1, 8, 0), (2, 4, 16, 16, 32, 0), (1, 2, 1, 37, 64, 36), (1, 2, 5, 40, 64, 35),
          (2, 3, 130, 130, 64, 0), (1, 14, 70, 200, 64, 130)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("blocks", [(16, 16), (7, 13), (64, 64)])
def test_tiled_matches_reference(shape, causal, blocks):
    B, H, Tq, Tk, D, off = shape
    if not causal and off:
        pytest.skip("offset only meaningful with a causal mask")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, H, t, D, generator=g) for t in (Tq, Tk, Tk))
    ref = reference_attention(q, k, v, q_offset=off, causal=causal)
    tiled = tiled_attention(q, k, v, q_offset=off, causal=causal, block_q=blocks[0], block_k=blocks[1])
    torch.testing.assert_close(tiled, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sdpa_attention(q, k, v, q_offset=off, causal=causal), ref, atol=1e-5, rtol=1e-5)


def test_large_logits_are_stable():
    """Scores around 1e4 overflow a naive exp(); both implementations subtract the running max."""
    q = torch.randn(1, 1, 32, 16) * 300
    k = torch.randn(1, 1, 32, 16) * 300
    v = torch.randn(1, 1, 32, 16)
    ref = reference_attention(q, k, v)
    assert torch.isfinite(ref).all()
    torch.testing.assert_close(tiled_attention(q, k, v, block_q=8, block_k=8), ref, atol=1e-5, rtol=1e-5)


def test_causal_mask_is_enforced():
    """Changing future keys/values must not change earlier outputs."""
    q, k, v = (torch.randn(1, 2, 20, 16) for _ in range(3))
    a = tiled_attention(q, k, v, block_q=4, block_k=4)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 10:] += 5
    v2[:, :, 10:] += 5
    b = tiled_attention(q, k2, v2, block_q=4, block_k=4)
    torch.testing.assert_close(a[:, :, :10], b[:, :, :10])
