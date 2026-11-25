# attention-variants-mla

[![CI](https://github.com/kishorhange111/attention-variants-mla/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/attention-variants-mla/actions/workflows/ci.yml)

MHA, GQA, MQA and DeepSeek-V2's **Multi-head Latent Attention (MLA)** from scratch, with
decoupled RoPE and the **absorbed-matrix decode path**, compared on the trade-off that
matters for serving: KV-cache bytes per token against language-modelling quality.

## Layers ([`attention.py`](src/attnx/attention.py))

| layer | cached per token per layer | elements |
|---|---|---|
| MHA | k, v for every head | 2·H·d_h |
| GQA | k, v per group of heads | 2·G·d_h |
| MQA | one shared k, v | 2·d_h |
| MLA | latent c_kv plus one shared RoPE key | d_c + d_r |

**MLA.** Keys and values are up-projections of a small per-token latent,
`c_kv = W_dkv·x`, with `k_i = W_uk,i·c_kv` and `v_i = W_uv,i·c_kv`. Only `c_kv` is cached.
RoPE would rotate the reconstructed keys by position, which would stop `W_uk` from being
folded into the query. So position is carried by a separate **decoupled RoPE key**
`k_r = RoPE(W_kr·x)` of width `d_r`, shared across heads, with a matching per-head `q_r`.

**Absorption.** `q_i · (W_uk,i c) = (W_uk,iᵀ q_i) · c`, and `Σ_j p_j W_uv,i c_j = W_uv,i Σ_j p_j c_j`.
At decode time each query is mapped into latent space once, scored directly against the
cached latents, and the attention-weighted latent is up-projected once.
`MLA.decode_absorbed` never materialises per-head keys or values.

## Tests

There are 13, on CPU. They check:
- **absorbed decoding equals the naive forward at every position** (float64, 1e-10);
- incremental latent caching with a RoPE offset equals computing all at once;
- MLA's content keys have rank ≤ d_c;
- RoPE scores depend only on relative position;
- GQA with all heads equals hand-written MHA;
- all four variants are causal, and their cache sizes match the table;
- the training script runs and resumes.

## Experiment ([`train.py`](src/attnx/train.py))

Four 6-layer, d = 384 byte-level GPTs train on Python standard-library source. They have
the same MLP, optimiser, schedule, 3,000 steps and 256-byte context; only the attention
layer differs. MLA uses d_c = 96, d_cq = 192 and d_r = 32. The script reports parameters,
KV-cache bytes per token in fp16 over all layers, the cache size at 32k tokens and val
bits/byte.

```
pytest -q                                          # 13 tests, CPU
python -m attnx.train --out results --device cuda  # resumable (runs.jsonl)
```

**Results are pending** from the Colab run.
