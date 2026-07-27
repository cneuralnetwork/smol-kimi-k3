from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from k3 import K3Config, K3ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a TinyStory from a smol-kimi-k3 checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tinystories/tokenizer.json"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = K3Config(**checkpoint["config"])
    model = K3ForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    input_ids = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.token_to_id("<eos>"),
    )
    print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
