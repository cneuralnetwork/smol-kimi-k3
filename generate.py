from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer

from k3 import K3Config, K3ForCausalLM


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
    config_path: Path | None = None,
) -> K3ForCausalLM:
    if checkpoint_path.suffix == ".safetensors":
        resolved_config = config_path or checkpoint_path.parent / "config.json"
        config = K3Config.from_json(resolved_config)
        model = K3ForCausalLM(config)
        load_model(model, checkpoint_path, strict=True)
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        config = K3Config(**checkpoint["config"])
        model = K3ForCausalLM(config)
        model.load_state_dict(checkpoint["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a TinyStory from a smol-kimi-k3 checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.checkpoint, device, args.config)
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = (
            args.checkpoint.parent / "tokenizer.json"
            if args.checkpoint.suffix == ".safetensors"
            else Path("data/tinystories/tokenizer.json")
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
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
