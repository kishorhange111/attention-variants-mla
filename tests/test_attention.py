import json

import pytest
import torch

from attnx.attention import GQA, MLA, apply_rope, make, rope_cache

D, H = 64, 4


def _x(l=12, b=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, l, D)


@pytest.mark.parametrize("kind", ["mha", "gqa2", "mqa", "mla"])
def test_shapes_and_causality(kind):
    torch.manual_seed(0)
    m = make(kind, D, H)
    x = _x()
    y = m(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 7:] += torch.randn_like(x2[:, 7:])  # change the future only
    torch.testing.assert_close(m(x2)[:, :7], y[:, :7], atol=1e-5, rtol=1e-5)


def test_cache_sizes():
    hd = D // H
    assert make("mha", D, H).kv_elems_per_token() == 2 * H * hd
    assert make("gqa2", D, H).kv_elems_per_token() == 2 * 2 * hd
    assert make("mqa", D, H).kv_elems_per_token() == 2 * hd
    assert make("mla", D, H).kv_elems_per_token() == D // 4 + hd // 2


def test_gqa_with_all_heads_equals_manual_mha():
    torch.manual_seed(0)
    m = GQA(D, H, H)
    x = _x()
    b, l, _ = x.shape
    hd = D // H
    q = m.q(x).view(b, l, H, hd).transpose(1, 2)
    k, v = m.kv(x).view(b, l, 2, H, hd).permute(2, 0, 3, 1, 4)
    cos, sin = rope_cache(l, hd)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (q @ k.transpose(-1, -2)) / hd**0.5
    s = s.masked_fill(torch.ones(l, l, dtype=torch.bool).triu(1), float("-inf"))
    ref = m.o((s.softmax(-1) @ v).transpose(1, 2).reshape(b, l, -1))
    torch.testing.assert_close(m(x), ref, atol=1e-5, rtol=1e-5)


def test_rope_is_relative():
    """q·k after RoPE depends only on the position difference."""
    torch.manual_seed(0)
    q, k = torch.randn(16), torch.randn(16)
    cos, sin = rope_cache(40, 16)

    def dot(i, j):
        return (apply_rope(q, cos[i], sin[i]) * apply_rope(k, cos[j], sin[j])).sum()

    torch.testing.assert_close(dot(5, 2), dot(30, 27), atol=1e-4, rtol=1e-4)


def test_mla_absorbed_decode_matches_naive_forward_at_every_position():
    torch.manual_seed(0)
    m = MLA(D, H, d_c=16, d_cq=32, d_r=8, head_dim=16).double()
    x = _x(l=10).double()
    full = m(x)
    c, kr = m.latents(x)
    for t in range(x.shape[1]):
        y = m.decode_absorbed(x[:, t : t + 1], c[:, : t + 1], kr[:, : t + 1])
        torch.testing.assert_close(y[:, 0], full[:, t], atol=1e-10, rtol=1e-10)


def test_mla_latents_with_offset_match_prefix_computation():
    """Caching incrementally (one token at a time with an offset) equals all at once."""
    torch.manual_seed(0)
    m = MLA(D, H, d_c=16, d_cq=32, d_r=8, head_dim=16).double()
    x = _x(l=6).double()
    c, kr = m.latents(x)
    for t in range(6):
        ct, krt = m.latents(x[:, t : t + 1], offset=t)
        torch.testing.assert_close(ct[:, 0], c[:, t])
        torch.testing.assert_close(krt[:, 0], kr[:, t])


def test_mla_without_rope_part_is_low_rank_attention():
    """The non-RoPE keys of all heads live in a d_c-dimensional subspace of the input."""
    torch.manual_seed(0)
    m = MLA(D, H, d_c=8, d_cq=32, d_r=8, head_dim=16)
    x = torch.randn(1, 50, D)
    k = m.w_uk(m.w_dkv(x))[0]  # (50, H*hd = 64)
    assert torch.linalg.matrix_rank(k).item() <= 8


def test_gradients_flow_through_mla():
    torch.manual_seed(0)
    m = make("mla", D, H)
    m(_x()).pow(2).mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_bad_group_count():
    with pytest.raises(ValueError):
        GQA(D, 4, 3)
    with pytest.raises(ValueError):
        make("nope", D, H)


def test_tiny_training_run(tmp_path):
    from attnx.train import main

    main(
        [
            "--out",
            str(tmp_path),
            "--device",
            "cpu",
            "--tiny",
            "--steps",
            "10",
            "--bs",
            "4",
            "--ctx",
            "32",
            "--kinds",
            "mqa,mla",
        ]
    )
    rows = [json.loads(s) for s in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["mqa", "mla"]
    assert all(0 < r["val_bpb"] < 9 for r in rows)
    main(["--out", str(tmp_path), "--device", "cpu", "--tiny", "--kinds", "mqa"])  # resumes
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2
