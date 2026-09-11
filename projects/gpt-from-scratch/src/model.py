from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

class GPT(nn.Module):
    def __init__(self, vocab_size: int, tokenizer: Tokenizer | None = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        self.tokenizer = tokenizer

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.token_embedding(tokens)
        if targets is None:
            return logits

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
        return tokens

    def generate_text(self, prompt: str, max_new_tokens: int) -> str:
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to generate text")
        tokens = self.tokenizer.encode(prompt).ids
        tokens = torch.tensor(tokens).unsqueeze(0)
        return self.tokenizer.decode(self.generate(tokens, max_new_tokens)[0].tolist())
    
