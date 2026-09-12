"""Data preparation for next-token language modeling."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer


def tokenize_file(
    source: str | Path,
    destination: str | Path,
    tokenizer: Tokenizer,
    *,
    chunk_size: int = 4 * 1024 * 1024,
) -> int:
    """Stream a UTF-8 corpus into a compact uint16 token cache."""
    if tokenizer.get_vocab_size() > 65_536:
        raise ValueError("uint16 token caches require a vocabulary of at most 65,536 tokens")

    source = Path(source)
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    total_bytes = source.stat().st_size
    characters_read = 0
    token_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with source.open(encoding="utf-8") as corpus, temporary.open("wb") as cache:
            remainder = ""
            while chunk := corpus.read(chunk_size):
                characters_read += len(chunk)
                chunk = remainder + chunk
                boundary = chunk.rfind("\n")
                if boundary == -1:
                    remainder = chunk
                    continue
                text, remainder = chunk[: boundary + 1], chunk[boundary + 1 :]
                token_ids = tokenizer.encode(text).ids
                array("H", token_ids).tofile(cache)
                token_count += len(token_ids)
                progress = min(100.0, characters_read / total_bytes * 100)
                print(f"tokenizing: {progress:5.1f}% ({token_count:,} tokens)", flush=True)

            if remainder:
                token_ids = tokenizer.encode(remainder).ids
                array("H", token_ids).tofile(cache)
                token_count += len(token_ids)

        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return token_count


@dataclass(frozen=True)
class LanguageModelDataset:
    """Tokenized train/validation splits with random contiguous batching."""

    tokenizer: Tokenizer
    train_tokens: torch.Tensor
    validation_tokens: torch.Tensor

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: Tokenizer,
        train_fraction: float = 0.9,
    ) -> "LanguageModelDataset":
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        token_ids = tokenizer.encode(text).ids
        if len(token_ids) < 2:
            raise ValueError("text must produce at least two tokens")

        tokens = torch.tensor(token_ids, dtype=torch.long)
        split_index = int(len(tokens) * train_fraction)
        if split_index == 0 or split_index == len(tokens):
            raise ValueError(
                "tokenized text is too short for the requested train/validation split"
            )
        return cls(tokenizer, tokens[:split_index], tokens[split_index:])

    @classmethod
    def from_token_file(
        cls,
        path: str | Path,
        tokenizer: Tokenizer,
        train_fraction: float = 0.9,
    ) -> "LanguageModelDataset":
        """Memory-map a uint16 token cache without loading it all into RAM."""
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        path = Path(path)
        if path.stat().st_size % 2:
            raise ValueError("invalid uint16 token cache size")
        token_count = path.stat().st_size // 2
        if token_count < 2:
            raise ValueError("token cache must contain at least two tokens")
        tokens = torch.from_file(
            str(path), shared=False, size=token_count, dtype=torch.uint16
        )
        split_index = int(token_count * train_fraction)
        return cls(tokenizer, tokens[:split_index], tokens[split_index:])

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        *,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size and context_length must be positive")

        tokens = self.train_tokens if split == "train" else self.validation_tokens
        maximum_start = len(tokens) - context_length
        if maximum_start <= 0:
            raise ValueError(f"{split} split must be longer than context_length")

        starts = torch.randint(maximum_start, (batch_size,), generator=generator)
        inputs = torch.stack([tokens[start : start + context_length] for start in starts])
        targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
        return inputs.to(device=device, dtype=torch.long), targets.to(
            device=device, dtype=torch.long
        )
