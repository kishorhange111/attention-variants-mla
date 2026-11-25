"""Four attention layers that trade KV-cache size against quality.

All take ``x: (b, l, d)`` and use causal attention with RoPE. KV-cache elements per token
per layer (×2 bytes in fp16):

| layer | cached per token | elements |
|---|---|---|
| MHA | k, v for every head | 2·H·d_h |
| GQA | k, v for each of G groups | 2·G·d_h |
| MQA | one shared k, v | 2·d_h |
| MLA (DeepSeek-V2) | latent c_kv plus one shared RoPE key | d_c + d_r |

**MLA.** Keys and values are up-projections of a small latent ``c_kv = W_dkv·x``:
``k_i = W_uk,i·c_kv`` and ``v_i = W_uv,i·c_kv``. RoPE can't be applied to keys
reconstructed from a cached latent without breaking the trick below, so position goes
through a separate *decoupled* RoPE key ``k_r = RoPE(W_kr·x)`` of width ``d_r``, shared by
all heads. Queries get a matching ``q_r`` per head. The score is
``(q_nope·k_nope + q_r·k_r) / √(d_h + d_r)``.

**Absorption** (decode). ``q_nope,i · (W_uk,i c) = (W_uk,iᵀ q_nope,i) · c``, so each query
can be mapped into latent space once and scored against the cached latents directly.
Likewise ``Σ_j p_j W_uv,i c_j = W_uv,i Σ_j p_j c_j``. Decoding therefore never
materialises per-head keys or values; it reads ``d_c + d_r`` numbers per cached token.
``MLA.decode_absorbed`` implements this, and the tests check that it equals the naive
forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def rope_cache(l: int, dim: int, device=None, base: float = 10000.0, offset: int = 0):
    inv = 1.0 / base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    ang = torch.outer(torch.arange(offset, offset + l, device=device, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    a, b = x.chunk(2, -1)
    return x * cos.to(x.dtype) + torch.cat([-b, a], -1) * sin.to(x.dtype)


class GQA(nn.Module):
    """Grouped-query attention; ``kv_heads == heads`` is MHA and ``kv_heads == 1`` is MQA."""

    def __init__(self, d: int, heads: int, kv_heads: int, head_dim: int | None = None):
        super().__init__()
        if heads % kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        self.h, self.g, self.hd = heads, kv_heads, head_dim or d // heads
        self.q = nn.Linear(d, heads * self.hd, bias=False)
        self.kv = nn.Linear(d, 2 * kv_heads * self.hd, bias=False)
        self.o = nn.Linear(heads * self.hd, d, bias=False)

    def kv_elems_per_token(self) -> int:
        return 2 * self.g * self.hd

    def forward(self, x):
        b, l, _ = x.shape
        q = self.q(x).view(b, l, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(b, l, 2, self.g, self.hd).permute(2, 0, 3, 1, 4)
        cos, sin = rope_cache(l, self.hd, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.g
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(b, l, -1))


class MLA(nn.Module):
    def __init__(self, d: int, heads: int, d_c: int, d_cq: int, d_r: int, head_dim: int):
        super().__init__()
        self.h, self.hd, self.dc, self.dr = heads, head_dim, d_c, d_r
        self.w_dq = nn.Linear(d, d_cq, bias=False)
        self.w_uq = nn.Linear(d_cq, heads * head_dim, bias=False)
        self.w_qr = nn.Linear(d_cq, heads * d_r, bias=False)
        self.w_dkv = nn.Linear(d, d_c, bias=False)
        self.w_uk = nn.Linear(d_c, heads * head_dim, bias=False)
        self.w_uv = nn.Linear(d_c, heads * head_dim, bias=False)
        self.w_kr = nn.Linear(d, d_r, bias=False)
        self.o = nn.Linear(heads * head_dim, d, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim + d_r)

    def kv_elems_per_token(self) -> int:
        return self.dc + self.dr

    def _q(self, x, cos, sin):
        b, l, _ = x.shape
        cq = self.w_dq(x)
        qn = self.w_uq(cq).view(b, l, self.h, self.hd).transpose(1, 2)
        qr = apply_rope(self.w_qr(cq).view(b, l, self.h, self.dr).transpose(1, 2), cos, sin)
        return qn, qr

    def latents(self, x, offset: int = 0):
        """What the cache stores per token: c_kv (b, l, d_c) and the RoPE key (b, l, d_r)."""
        cos, sin = rope_cache(x.shape[1], self.dr, x.device, offset=offset)
        return self.w_dkv(x), apply_rope(self.w_kr(x), cos, sin)

    def forward(self, x):
        """Naive (training/prefill) path: materialise per-head keys and values."""
        b, l, _ = x.shape
        cos, sin = rope_cache(l, self.dr, x.device)
        qn, qr = self._q(x, cos, sin)
        c, kr = self.latents(x)
        kn = self.w_uk(c).view(b, l, self.h, self.hd).transpose(1, 2)
        v = self.w_uv(c).view(b, l, self.h, self.hd).transpose(1, 2)
        k = torch.cat([kn, kr[:, None].expand(b, self.h, l, self.dr)], -1)
        q = torch.cat([qn, qr], -1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.o(y.transpose(1, 2).reshape(b, l, -1))

    def decode_absorbed(self, x_t, c_cache, kr_cache):
        """One new token ``x_t: (b, 1, d)`` attending to cached latents *including itself*
        (``c_cache: (b, T, d_c)``, ``kr_cache: (b, T, d_r)``), without per-head k or v."""
        b, T = c_cache.shape[:2]
        cos, sin = rope_cache(1, self.dr, x_t.device, offset=T - 1)
        qn, qr = self._q(x_t, cos, sin)  # (b, h, 1, hd), (b, h, 1, d_r)
        w_uk = self.w_uk.weight.view(self.h, self.hd, self.dc)  # k_i = W_uk,i c
        q_lat = torch.einsum("bhqe,hec->bhqc", qn, w_uk)  # W_uk,iᵀ q_nope,i
        s = torch.einsum("bhqc,btc->bhqt", q_lat, c_cache)
        s = s + torch.einsum("bhqr,btr->bhqt", qr, kr_cache)
        p = (s * self.scale).softmax(-1)
        ctx = torch.einsum("bhqt,btc->bhqc", p, c_cache)  # attend in latent space
        w_uv = self.w_uv.weight.view(self.h, self.hd, self.dc)
        y = torch.einsum("bhqc,hec->bhqe", ctx, w_uv)  # then up-project once
        return self.o(y.transpose(1, 2).reshape(b, 1, -1))


def make(kind: str, d: int, heads: int, **kw) -> nn.Module:
    if kind == "mha":
        return GQA(d, heads, heads)
    if kind == "mqa":
        return GQA(d, heads, 1)
    if kind.startswith("gqa"):
        return GQA(d, heads, int(kind[3:]))
    if kind == "mla" or kind.startswith("mla-c"):
        # "mla" uses d_c = d/4; "mla-c192" sets the latent width explicitly
        hd = d // heads
        d_c = int(kind[5:]) if kind.startswith("mla-c") else kw.get("d_c", d // 4)
        return MLA(
            d,
            heads,
            d_c=d_c,
            d_cq=kw.get("d_cq", d // 2),
            d_r=kw.get("d_r", hd // 2),
            head_dim=hd,
        )
    raise ValueError(kind)
