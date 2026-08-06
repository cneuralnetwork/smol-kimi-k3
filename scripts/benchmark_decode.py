#!/usr/bin/env python3
"""Benchmark autoregressive decode speed (forward_step) for verifying the
vectorized-MoE speedup.

Runs 200 single-token forward_steps after a prompt prefill, reports tok/s
and ms/step. Typical numbers on RTX 3090 with the released 49M checkpoint:

    before perf/vectorized-moe-forward: ~16 tok/s (61 ms/step)
    after  perf/vectorized-moe-forward: ~35 tok/s (29 ms/step)  -> ~2.1x

Usage:
    python scripts/benchmark_decode.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from k3 import K3Config, K3ForCausalLM  # noqa: E402
from safetensors.torch import load_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "pretrained/smol-kimi-k3/model.safetensors"))
    p.add_argument("--config", default=str(REPO_ROOT / "pretrained/smol-kimi-k3/config.json"))
    p.add_argument("--tokenizer", default=str(REPO_ROOT / "pretrained/smol-kimi-k3/tokenizer.json"))
    p.add_argument("--prompt", default="Once upon a time there was a little bird")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = K3Config.from_json(args.config)
    model = K3ForCausalLM(cfg)
    load_model(model, args.checkpoint, strict=True)
    model = model.to(device).eval()

    tok = Tokenizer.from_file(args.tokenizer)
    pt = torch.tensor([tok.encode(args.prompt).ids], device=device)

    with torch.inference_mode():
        cache = None
        logits = None
        for pos in range(pt.shape[1]):
            logits, cache = model.forward_step(pt[:, pos:pos + 1], cache)
        for _ in range(args.warmup):
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            logits, cache = model.forward_step(nxt, cache)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.n_steps):
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            logits, cache = model.forward_step(nxt, cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    tok_s = args.n_steps / elapsed
    ms_per_step = elapsed * 1000 / args.n_steps
    print(f"{tok_s:.2f} tok/s   {ms_per_step:.2f} ms/step   ({args.n_steps} steps in {elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
