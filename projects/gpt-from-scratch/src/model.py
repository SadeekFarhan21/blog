from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


## Single head self attention
class SelfAttention(nn.Module):
    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.head_size = head_size
        self.n_embed = n_embed
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        ## compare every token with every other token
        scores = queries @ keys.transpose(-2, -1)

        ## scale scores for numerical stability
        scores = scores * (self.head_size ** -0.5)

        ## unmasked self-attention
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        ## combine value vectors according to the attention weights
        return weights @ values

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
        self.attention = SelfAttention(head_size=n_embed, n_embed=n_embed, block_size=block_size, dropout=0.0)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length = tokens.shape
        if sequence_length > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {sequence_length}, block size is only {self.block_size}")
        ## Position embeddings
        position = self.position_embedding(torch.arange(0, sequence_length, dtype=torch.long, device=tokens.device))
        ## Token embeddings
        token = self.token_embedding(tokens)
        x = token + position
        ## Unmasked self-attention
        x = self.attention(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Each position predicts the token at the corresponding position
            # in targets (the dataset has already shifted targets by one).
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return loss

        return logits

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size:]
            ## get the logits for the last position
            logits = self.forward(context)
            ## get the last position logits
            logits = logits[:, -1, :]
            ## sample the next token
            probs = F.softmax(logits, dim=-1)
            ## sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            ## append the next token to the tokens
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def generate_text(self, prompt: str, max_new_tokens: int) -> str:
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to generate text")
        token_ids = self.tokenizer.encode(prompt).ids
        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        generated = self.generate(tokens, max_new_tokens)
        return self.tokenizer.decode(generated[0].tolist())

    ## At this point we have a model that is bigram model which means that the model predicts the next token based on only the previous token, but we want to take it to the next level and make it a transformer model.

    
