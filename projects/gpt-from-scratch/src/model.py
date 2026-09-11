from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)
    
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
        return tokens
    
    def generate_text(self, prompt: str, max_new_tokens: int) -> str:
        tokens = self.tokenizer.encode(prompt).ids
        tokens = torch.tensor(tokens).unsqueeze(0)
        return self.tokenizer.decode(self.generate(tokens, max_new_tokens)[0].tolist())
    