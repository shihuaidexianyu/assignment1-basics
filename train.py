from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics import train_bpe


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _encode_token_bytes(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token_bytes)


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_dir: Path,
    vocab_filename: str = "tokenizer_vocab.json",
    merges_filename: str = "tokenizer_merges.txt",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    byte_encoder = _bytes_to_unicode()

    vocab_out = {_encode_token_bytes(token_bytes, byte_encoder): token_id for token_id, token_bytes in vocab.items()}
    vocab_path = out_dir / vocab_filename
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)

    merges_path = out_dir / merges_filename
    with merges_path.open("w", encoding="utf-8") as f:
        for token_a, token_b in merges:
            token_a_str = _encode_token_bytes(token_a, byte_encoder)
            token_b_str = _encode_token_bytes(token_b, byte_encoder)
            f.write(f"{token_a_str} {token_b_str}\n")

    return vocab_path, merges_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer and save vocab/merges.")
    parser.add_argument(
        "--input",
        default="/home/hw/learn/assignment1-basics/data/data/TinyStoriesV2-GPT4-train.txt",
        help="Path to training corpus (text file).",
    )
    parser.add_argument("--vocab-size", type=int, default=1000, help="Total vocabulary size including special tokens.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to reserve (repeatable).",
    )
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers for pre-tokenization.")
    parser.add_argument(
        "--out-dir", default="/home/hw/learn/assignment1-basics/tokenizer_out", help="Output directory."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab, merges = train_bpe.train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        num_workers=args.num_workers,
    )
    vocab_path, merges_path = save_tokenizer(vocab, merges, Path(args.out_dir))
    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")


if __name__ == "__main__":
    main()
