"""Train one small byte-level GPT per attention variant and compare quality against cache size.

    python -m attnx.train --out results --device cuda

Every variant gets the same data (Python standard-library source as bytes), width, depth,
MLP, optimiser, schedule and steps; only the attention layer changes. Reported per variant:
parameters, KV-cache bytes per token (fp16, all layers), cache size at 32k tokens, and val
bits/byte. Each run is appended to ``<out>/runs.jsonl`` as it finishes, so a reclaimed
session skips finished variants.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attention import make

LN2 = math.log(2)


class Block(nn.Module):
    def __init__(self, kind, d, heads, **kw):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = make(kind, d, heads, **kw)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, kind, d=384, heads=6, layers=6, **kw):
        super().__init__()
        self.emb = nn.Embedding(256, d)
        self.blocks = nn.ModuleList(Block(kind, d, heads, **kw) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, 256, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def kv_bytes_per_token(self) -> int:
        return 2 * sum(b.attn.kv_elems_per_token() for b in self.blocks)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def load_corpus():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < 50 else tr).append(fh.read())
    return (
        np.frombuffer(b"\n".join(tr), np.uint8).copy(),
        np.frombuffer(b"\n".join(va), np.uint8).copy(),
    )


def batch(data, bs, l, rng, device):
    ix = rng.integers(0, len(data) - l - 1, bs)
    x = torch.from_numpy(np.stack([data[i : i + l + 1] for i in ix]).astype(np.int64))
    x = x.to(device)
    return x[:, :-1], x[:, 1:]


def run(kind, data, val, steps, bs, ctx, lr, device, seed=0, log=print, **model_kw):
    torch.manual_seed(seed)
    model = GPT(kind, **model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        f = min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.02)
        x, y = batch(data, bs, ctx, rng, device)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, steps // 5) == 0:
            log(f"  {kind} step {step}: {loss.item() / LN2:.3f} bits/byte")
    model.eval()
    vr = np.random.default_rng(123)
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
        losses = []
        for _ in range(20):
            x, y = batch(val, bs, ctx, vr, device)
            losses.append(F.cross_entropy(model(x).float().flatten(0, 1), y.flatten()).item())
    kvb = model.kv_bytes_per_token()
    return {
        "kind": kind,
        "params": sum(p.numel() for p in model.parameters()),
        "kv_bytes_per_token": kvb,
        "kv_mib_at_32k": kvb * 32768 / 2**20,
        "val_bpb": float(np.mean(losses)) / LN2,
        "minutes": (time.time() - t0) / 60,
        "steps": steps,
    }


def report(rows) -> str:
    base = next((r for r in rows if r["kind"] == "mha"), None)
    out = [
        "| attention | params | KV bytes/token | KV at 32k tokens | vs MHA | val bits/byte |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rel = f"{r['kv_bytes_per_token'] / base['kv_bytes_per_token']:.3f}×" if base else "—"
        out.append(
            f"| {r['kind']} | {r['params'] / 1e6:.2f}M | {r['kv_bytes_per_token']:,} | "
            f"{r['kv_mib_at_32k']:.1f} MiB | {rel} | {r['val_bpb']:.3f} |"
        )
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kinds", default="mha,gqa2,mqa,mla")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    path = os.path.join(a.out, "runs.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done[r["kind"]] = r
    data, val = load_corpus()
    kw = dict(d=32, heads=4, layers=1) if a.tiny else {}
    for kind in a.kinds.split(","):
        if kind in done:
            log(f"skip {kind}")
            continue
        r = run(kind, data, val, a.steps, a.bs, a.ctx, a.lr, a.device, log=log, **kw)
        r["device"] = torch.cuda.get_device_name() if a.device == "cuda" else "cpu"
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        done[kind] = r
        log(json.dumps(r))
    text = report([done[k] for k in a.kinds.split(",") if k in done])
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
