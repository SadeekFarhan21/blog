from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embed: int, tokenizer: Tokenizer | None = None) -> None:
        super().__init__()
        # Bigram model
        # self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        # Transformer model
        self.block_size = block_size ## the context length of the model
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.tokenizer = tokenizer

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length = tokens.shape
        if sequence_length > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {sequence_length}, block size is only {self.block_size}")
        ## Position embeddings
        position = self.position_embedding(torch.arange(0, sequence_length, dtype=torch.long, device=tokens.device))
        ## Token embeddings
        token = self.token_embedding(tokens)
        x = token + position
        ## Transformer
        x = self.transformer(x)
        return x

    def generate_text(self, prompt: str, max_new_tokens: int) -> str:
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to generate text")
        tokens = self.tokenizer.encode(prompt).ids
        tokens = torch.tensor(tokens).unsqueeze(0)
        return self.tokenizer.decode(self.generate(tokens, max_new_tokens)[0].tolist())
    
    ## At this point we have a model that is bigram model which means that the model predicts the next token based on only the previous token, but we want to take it to the next level and make it a transformer model.
