"""A from-scratch Qwen2 decoder forward pass with an explicit, pre-allocated KV cache.

Only the weights are taken from the Hugging Face model; embedding lookup, RMSNorm,
rotary position embeddings, grouped-query attention, the SwiGLU MLP, causal masking
and cache management are implemented here. The attention kernel is pluggable
(reference / tiled online-softmax / PyTorch SDPA, see attention.py).
"""
import torch

from .attention import ATTENTION


class KVCache:
    """Per-layer key/value tensors of shape [B, n_kv_heads, max_len, head_dim].

    `length` is the number of positions already stored; new keys/values are written
    at [length, length + T). Rolling back (for rejected speculative tokens) only
    lowers `length`: stale entries beyond it are never read and get overwritten.
    """

    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, dtype, device):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0
        self.max_len = max_len

    def write(self, layer, k, v):
        """Store k/v for new positions and return the full (past + new) keys/values for this layer."""
        start, end = self.length, self.length + k.shape[2]
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: need {end} positions, allocated {self.max_len}")
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n):
        self.length += n

    def rollback(self, length):
        assert 0 <= length <= self.length
        self.length = length

    def nbytes(self):
        """Bytes currently in use (not allocated): 2 tensors x layers x B x kv_heads x length x head_dim."""
        L, B, H, _, D = self.k.shape
        return 2 * L * B * H * self.length * D * self.k.element_size()


def _rope_theta(config):
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = config.rope_parameters["rope_theta"]
    return float(theta)


class ManualQwen2:
    def __init__(self, hf_model, attention="reference"):
        self.cfg = cfg = hf_model.config
        self.m = hf_model.model
        self.lm_head = hf_model.lm_head
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype
        self.set_attention(attention)
        # RoPE frequencies: theta^(-2i/d) for i in [0, d/2).
        d = self.head_dim
        self.inv_freq = 1.0 / (_rope_theta(cfg) ** (torch.arange(0, d, 2, dtype=torch.float32, device=self.device) / d))

    def set_attention(self, name):
        self.attention_name = name
        self.attn = ATTENTION[name]

    def new_cache(self, max_len, batch=1):
        return KVCache(self.n_layers, batch, self.n_kv, max_len, self.head_dim, self.dtype, self.device)

    def _rms_norm(self, x, weight):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return weight * xf.to(x.dtype)

    def _rope(self, positions):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # [T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, d]
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _apply_rope(x, cos, sin):
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin

    @torch.no_grad()
    def forward(self, input_ids, cache=None, last_only=False, select=None):
        """Run new tokens through the decoder.

        input_ids: [B, T] tokens at absolute positions cache.length .. cache.length + T - 1
        (0 .. T-1 without a cache, i.e. the uncached reference path recomputes everything).
        Returns logits [B, T, V] (or [B, 1, V] with last_only, or [B, 1, V] at per-row
        positions `select` [B]) and advances the cache by T.
        """
        B, T = input_ids.shape
        offset = cache.length if cache is not None else 0
        positions = torch.arange(offset, offset + T, device=self.device)
        cos, sin = self._rope(positions)

        h = self.m.embed_tokens(input_ids)
        for i, layer in enumerate(self.m.layers):
            a = layer.self_attn
            x = self._rms_norm(h, layer.input_layernorm.weight)
            q = a.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
            if cache is not None:
                k, v = cache.write(i, k, v)  # keys/values for positions 0 .. offset + T - 1
            # Grouped-query attention: each KV head serves n_heads / n_kv query heads.
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            o = self.attn(q, k, v, q_offset=offset, causal=True)
            h = h + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))

            x = self._rms_norm(h, layer.post_attention_layernorm.weight)
            mlp = layer.mlp
            h = h + mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

        if cache is not None:
            cache.advance(T)
        if select is not None:
            h = h[torch.arange(B, device=self.device), select][:, None]
        elif last_only:
            h = h[:, -1:]
        h = self._rms_norm(h, self.m.norm.weight)
        return self.lm_head(h).float()

    @torch.no_grad()
    def decode_step_ragged(self, tokens, cache, lengths):
        """One decoding step for B independent sequences whose caches have different lengths.

        tokens: [B] next input token per row; lengths: [B] long tensor, positions already
        cached per row. Row b's token sits at position lengths[b]: its key/value is written
        there, RoPE uses that position, and it attends to keys 0..lengths[b] of its own row
        only (entries beyond a row's length are padding/stale and are masked out).
        Returns logits [B, V]; the caller advances `lengths`.
        """
        B = tokens.shape[0]
        rows = torch.arange(B, device=self.device)
        cos, sin = self._rope(lengths)  # [B, D]
        cos, sin = cos[:, None, None, :], sin[:, None, None, :]
        tk = int(lengths.max()) + 1
        mask = (torch.arange(tk, device=self.device)[None, :] <= lengths[:, None])[:, None, None, :]  # [B,1,1,Tk]

        h = self.m.embed_tokens(tokens[:, None])  # [B, 1, hidden]
        for i, layer in enumerate(self.m.layers):
            a = layer.self_attn
            x = self._rms_norm(h, layer.input_layernorm.weight)
            q = a.q_proj(x).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(B, 1, self.n_kv, self.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(B, 1, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
            cache.k[i][rows, :, lengths] = k[:, :, 0]  # scatter each row's new key to its own position
            cache.v[i][rows, :, lengths] = v[:, :, 0]
            rep = self.n_heads // self.n_kv
            keys = cache.k[i][:, :, :tk].repeat_interleave(rep, dim=1)
            vals = cache.v[i][:, :, :tk].repeat_interleave(rep, dim=1)
            o = self.attn(q, keys, vals, causal=False, mask=mask)
            h = h + a.o_proj(o.transpose(1, 2).reshape(B, 1, -1))
            x = self._rms_norm(h, layer.post_attention_layernorm.weight)
            mlp = layer.mlp
            h = h + mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        h = self._rms_norm(h, self.m.norm.weight)
        return self.lm_head(h)[:, 0].float()
