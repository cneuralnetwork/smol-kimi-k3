from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from k3 import K3Config, K3ForCausalLM
from k3.monitoring import TrainingMonitor
from k3.optim import K3Optimizers


def get_batch(
    data: np.memmap,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))
    sequences = [
        torch.from_numpy(
            np.asarray(data[position : position + sequence_length + 1], dtype=np.int64)
        )
        for position in positions.tolist()
    ]
    batch = torch.stack(sequences)
    return (
        batch[:, :-1].to(device, non_blocking=True),
        batch[:, 1:].to(device, non_blocking=True),
    )


def lr_scale(step: int, total_steps: int, warmup_fraction: float, min_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


@torch.no_grad()
def evaluate(
    model: K3ForCausalLM,
    data: np.memmap,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        input_ids, labels = get_batch(data, batch_size, sequence_length, device)
        with autocast_context():
            output = model(input_ids, labels)
        losses.append(output.loss.detach().float())
    model.train()
    return torch.stack(losses).mean().item()


def save_checkpoint(
    path: Path,
    model: K3ForCausalLM,
    optimizers: K3Optimizers,
    step: int,
    best_validation_loss: float,
    train_args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": optimizers.state_dict(),
            "config": model.config.to_dict(),
            "step": step,
            "best_validation_loss": best_validation_loss,
            "train_args": train_args,
        },
        temporary,
    )
    temporary.replace(path)


@torch.inference_mode()
def generate_sample(
    model: K3ForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    autocast_context,
) -> str:
    was_training = model.training
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids],
        dtype=torch.long,
        device=device,
    )
    with autocast_context():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            eos_token_id=tokenizer.token_to_id("<eos>"),
        )
    if was_training:
        model.train()
    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train smol-kimi-k3 on TinyStories"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/k3_50m.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/k3-50m"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--muon-lr", type=float, default=0.01)
    parser.add_argument("--adamw-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument("--sample-prompt", default="Once upon a time, a little fox")
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Verify the NVIDIA driver with nvidia-smi; "
            "this 50M configuration is intended for the local 8 GB GPU."
        )
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    config = K3Config.from_json(args.config)
    with (args.data_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["vocab_size"] != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab {metadata['vocab_size']} does not match model "
            f"vocab {config.vocab_size}"
        )
    train_data = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    validation_data = np.memmap(
        args.data_dir / "validation.bin", dtype=np.uint16, mode="r"
    )

    model = K3ForCausalLM(config).to(device)
    model_parameters = model.num_parameters()
    print(f"parameters: {model_parameters:,}")
    optimizers = K3Optimizers(
        model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizers.load_state_dict(checkpoint["optimizers"])
        start_step = checkpoint["step"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]
        print(f"resumed from step {start_step}")

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))
    monitor = TrainingMonitor(args.output_dir, model_parameters, args.steps)
    train_args = vars(args).copy()
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in train_args.items()
    }
    model.train()
    optimizers.zero_grad()
    last_time = time.perf_counter()
    tokens_since_log = 0

    def request_stop(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)
    try:
        for step in range(start_step, args.steps):
            scale = lr_scale(
                step,
                args.steps,
                args.warmup_fraction,
                args.minimum_lr_ratio,
            )
            optimizers.set_lr_scale(scale)
            accumulated_loss = 0.0
            for _ in range(args.gradient_accumulation):
                input_ids, labels = get_batch(
                    train_data,
                    args.batch_size,
                    args.sequence_length,
                    device,
                )
                with autocast_context():
                    output = model(input_ids, labels)
                    loss = output.loss / args.gradient_accumulation
                loss.backward()
                accumulated_loss += loss.detach().float().item()
                tokens_since_log += args.batch_size * args.sequence_length

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers.step()
            optimizers.zero_grad()

            if step % args.log_interval == 0:
                now = time.perf_counter()
                elapsed = now - last_time
                throughput = tokens_since_log / elapsed
                vram_gib = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"step {step:6d} | train loss {accumulated_loss:.4f} | "
                    f"lr scale {scale:.4f} | {throughput:,.0f} tok/s | "
                    f"vram {vram_gib:.2f} GiB"
                )
                monitor.record_train(
                    step,
                    accumulated_loss,
                    scale,
                    throughput,
                    vram_gib,
                )
                last_time = now
                tokens_since_log = 0

            if step % args.eval_interval == 0 or step == args.steps - 1:
                validation_loss = evaluate(
                    model,
                    validation_data,
                    args.eval_batches,
                    args.batch_size,
                    args.sequence_length,
                    device,
                    autocast_context,
                )
                print(f"step {step:6d} | validation loss {validation_loss:.4f}")
                monitor.record_validation(step, validation_loss)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizers,
                        step,
                        best_validation_loss,
                        train_args,
                    )

            should_sample = (
                not args.skip_samples
                and (step % args.sample_interval == 0 or step == args.steps - 1)
            )
            if should_sample:
                try:
                    sample_model = getattr(model, "_orig_mod", model)
                    sample = generate_sample(
                        sample_model,
                        tokenizer,
                        args.sample_prompt,
                        args.sample_tokens,
                        device,
                        autocast_context,
                    )
                    monitor.record_sample(step, args.sample_prompt, sample)
                    print(f"step {step:6d} | sample: {sample[:160]!r}")
                except Exception as error:
                    monitor.set_status(
                        "running",
                        f"Sample generation failed at step {step}: {error}",
                    )
                    print(f"step {step:6d} | sample generation failed: {error}")

            if step % args.save_interval == 0 or step == args.steps - 1:
                save_checkpoint(
                    args.output_dir / "latest.pt",
                    model,
                    optimizers,
                    step,
                    best_validation_loss,
                    train_args,
                )
        monitor.set_status("complete")
    except KeyboardInterrupt:
        monitor.set_status("stopped", "Training was stopped by the user.")
        print("training stopped")
    except Exception as error:
        monitor.set_status("error", str(error))
        raise


if __name__ == "__main__":
    main()
