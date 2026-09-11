"""Train a byte-level BPE tokenizer for the GPT training corpus."""

from __future__ import annotations
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
    files: Sequence[str | Path],
    *,
    vocab_size: int = 5_000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a tokenizer from one or more UTF-8 corpus files."""
    if not files:
        raise ValueError("at least one training file is required")

    paths = [Path(file) for file in files]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"training files not found: {', '.join(missing_paths)}")

    tokenizer = create_tokenizer()
    tokenizer.train(
        [str(path) for path in paths],
        trainer=create_trainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        ),
    )
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a previously saved tokenizer.json file."""
    return Tokenizer.from_file(str(path))
