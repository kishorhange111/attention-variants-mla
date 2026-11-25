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

## Results (Tesla T4, Colab)

There are 6 heads of width 64 and 6 layers, trained for 3,000 steps each, which takes 3–4
minutes per model. The table and raw rows are in [`results/`](results/).

| attention | params | KV bytes/token | KV at 32k tokens | vs MHA | val bits/byte |
|---|---|---|---|---|---|
| MHA | 10.82M | 9,216 | 288 MiB | 1.000× | 1.253 |
| GQA, 2 groups | 9.64M | 3,072 | 96 MiB | 0.333× | **1.254** |
| MQA | 9.34M | 1,536 | 48 MiB | 0.167× | 1.271 |
| MLA (d_c = 96, d_r = 32) | 10.01M | 1,536 | 48 MiB | 0.167× | 1.338 |

- **GQA with 2 groups matches MHA at a third of the cache** (1.254 vs 1.253 bits/byte).
- **MQA cuts the cache 6× and costs 0.018 bits/byte.**
- **MLA, at the same 1,536 bytes per token as MQA, was clearly worse here** (1.338 vs
  1.271). This does not reproduce DeepSeek-V2's claim that MLA beats MHA, and this setup
  likely explains why:
  - The latent is very narrow. d_c = 96 must hold the keys and values of all 6 heads,
    which take 768 dimensions in MHA. That is an 8× compression. For comparison,
    DeepSeek-V2 uses d_c = 512 for 128 heads of width 128.
  - Every variant used the same learning rate (1e-3), tuned for none of them. MLA adds two
    extra low-rank factorisations, which may want a different one.
  - These are small models trained for 3,000 steps, from one seed each.

  The follow-up below sweeps d_c and tries a lower learning rate. The absorbed decode path
  is verified exact (float64, 1e-10) either way. What is in question is the quality at this
  compression, not the implementation.

### Follow-up: latent width and learning rate

These are the same model and steps, again on a T4, with MQA rerun as a control. MQA's
1.269, against 1.271 in the first run, gives a rough idea of run-to-run noise. The raw rows
are in [`results/mla_dc/`](results/mla_dc/) and [`results/mla_lr3e-4/`](results/mla_lr3e-4/).

| attention | d_c | KV bytes/token | vs MHA | val bits/byte |
|---|---|---|---|---|
| MQA (control) | – | 1,536 | 0.167× | 1.269 |
| MLA | 64 | 1,152 | 0.125× | 1.345 |
| MLA (first run) | 96 | 1,536 | 0.167× | 1.338 |
| MLA | 192 | 2,688 | 0.292× | 1.273 |
| MLA | 352 | 4,608 | 0.500× | 1.267 |

| lr | MQA | MLA (d_c = 96) | gap |
|---|---|---|---|
| 1e-3 | 1.271 | 1.338 | 0.067 |
| 3e-4 | 1.349 | 1.475 | 0.126 |

- **Most of MLA's deficit is latent width.** Widening d_c from 96 to 192 recovers almost all
  of the gap: 1.273, level with MQA. Below about 100 dimensions quality is flat and poor
  (1.345 at 64, 1.338 at 96).
- **But here MLA never beats the grouped baselines at equal cache.**
  - At d_c = 192 it matches MQA while caching 1.75× more.
  - At d_c = 352 (1.267) it caches 1.5× more than GQA-2 and is still worse (1.254).
  - On this 10M-parameter, 6-head model, GQA is the better cache/quality trade.
  - DeepSeek-V2's result is at 128 heads, where one shared latent replaces far more
    per-head state. This experiment is too small to test that regime.
- **A lower learning rate does not help MLA.** At 3e-4 both get worse, and MLA's gap doubles
  (0.126). Only one value below 1e-3 was tried, so a higher rate is untested.
