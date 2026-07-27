from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def story_stream(split: str, limit: int) -> object:
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    records = dataset if limit <= 0 else islice(dataset, limit)
    for record in records:
        text = record["text"].strip()
        if text:
            yield text


def train_tokenizer(output_path: Path, vocab_size: int, sample_stories: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        story_stream("train", sample_stories),
        trainer=trainer,
        length=sample_stories if sample_stories > 0 else None,
    )
    tokenizer.save(str(output_path))
    return tokenizer


def encode_split(
    tokenizer: Tokenizer,
    split: str,
    output_path: Path,
    story_limit: int,
) -> dict:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("Tokenizer is missing <eos>")
    token_count = 0
    story_count = 0
    with output_path.open("wb") as handle:
        for text in tqdm(story_stream(split, story_limit), desc=f"encoding {split}"):
            ids = tokenizer.encode(text).ids + [eos_id]
            np.asarray(ids, dtype=np.uint16).tofile(handle)
            token_count += len(ids)
            story_count += 1
    return {"stories": story_count, "tokens": token_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories for local K3 training")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-stories", type=int, default=100_000)
    parser.add_argument(
        "--train-stories",
        type=int,
        default=0,
        help="0 uses the full training split",
    )
    parser.add_argument(
        "--validation-stories",
        type=int,
        default=10_000,
        help="0 uses the full validation split",
    )
    args = parser.parse_args()
    if args.vocab_size >= np.iinfo(np.uint16).max:
        raise ValueError("vocab-size must fit in uint16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    tokenizer = train_tokenizer(
        tokenizer_path,
        args.vocab_size,
        args.tokenizer_stories,
    )
    metadata = {
        "dataset": "roneneldan/TinyStories",
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "train": encode_split(
            tokenizer,
            "train",
            args.output_dir / "train.bin",
            args.train_stories,
        ),
        "validation": encode_split(
            tokenizer,
            "validation",
            args.output_dir / "validation.bin",
            args.validation_stories,
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
