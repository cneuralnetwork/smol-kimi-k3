from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import save_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k3 import K3Config, K3ForCausalLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a smol-kimi-k3 training checkpoint for inference"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release/smol-kimi-k3-tinystories"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tinystories/tokenizer.json"),
    )
    parser.add_argument("--dashboard-state", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = K3Config(**checkpoint["config"])
    model = K3ForCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(dtype=torch.bfloat16).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "model.safetensors"
    config_path = args.output_dir / "config.json"
    tokenizer_path = args.output_dir / "tokenizer.json"
    metadata_path = args.output_dir / "training_metadata.json"
    checksums_path = args.output_dir / "sha256sums.txt"

    save_model(
        model,
        weights_path,
        metadata={
            "model": "smol-kimi-k3",
            "format": "pt",
            "dtype": "bfloat16",
            "checkpoint_step": str(checkpoint["step"]),
        },
    )
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(args.tokenizer, tokenizer_path)

    metadata = {
        "model": "smol-kimi-k3",
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "parameter_count": model.num_parameters(),
        "weight_dtype": "bfloat16",
        "exported_at": datetime.now(UTC).isoformat(),
        "training_args": checkpoint["train_args"],
    }
    if args.dashboard_state and args.dashboard_state.exists():
        metadata["final_run_state"] = json.loads(
            args.dashboard_state.read_text(encoding="utf-8")
        )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    packaged_files = [
        weights_path,
        config_path,
        tokenizer_path,
        metadata_path,
    ]
    checksums_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in packaged_files),
        encoding="utf-8",
    )
    for path in [*packaged_files, checksums_path]:
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
