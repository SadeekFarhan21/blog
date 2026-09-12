"""Train a byte-level BPE tokenizer for the GPT training corpus."""

from __future__ import annotations
import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


UNKNOWN_TOKEN = "<|unk|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS = [UNKNOWN_TOKEN, END_OF_TEXT_TOKEN]


def create_tokenizer() -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer."""
    tokenizer = Tokenizer(models.BPE(unk_token=UNKNOWN_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def create_trainer(
    *,
    vocab_size: int = 5_000,
    min_frequency: int = 2,
) -> trainers.BpeTrainer:
    """Create the trainer that learns BPE merges from a corpus."""
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must leave room for non-special tokens")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")

    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )


def train_from_iterator(
    texts: Iterable[str],
    *,
    vocab_size: int = 5_000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a tokenizer from an iterable of text samples."""
    tokenizer = create_tokenizer()
    tokenizer.train_from_iterator(
        texts,
        trainer=create_trainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        ),
    )
    return tokenizer


def train_from_files(
    files: str | Path | Sequence[str | Path],
    *,
    vocab_size: int = 5_000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a tokenizer from one or more UTF-8 corpus files."""
    if isinstance(files, (str, Path)):
        files = [files]

    tokenizer = create_tokenizer()
    tokenizer.train(
        [str(path) for path in files],
        trainer=create_trainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        ),
    )
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a previously saved tokenizer.json file."""
    return Tokenizer.from_file(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a saved tokenizer")
    parser.add_argument("tokenizer", type=Path, help="path to tokenizer.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="UTF-8 text file to encode")
    source.add_argument("--text", nargs="+", help="text to encode")
    args = parser.parse_args()
    if args.file:
        tokenizer = train_from_files(args.file)
    else:
        tokenizer = train_from_iterator([" ".join(args.text)])
    print("Tokenizing text...")
    print("Saving tokenizer...")
    tokenizer.save(str(args.tokenizer))
    print("Tokenizer saved to", args.tokenizer)
    print("Inspecting tokenizer...")
