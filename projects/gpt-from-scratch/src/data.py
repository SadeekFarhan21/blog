"""Data preparation for next-token language modeling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenizers import Tokenizer


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
        return inputs.to(device), targets.to(device)
