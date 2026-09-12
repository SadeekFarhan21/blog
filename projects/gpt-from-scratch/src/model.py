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

        ## causal mask to prevent tokens from attending to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        ## compare every token with every other token
        scores = queries @ keys.transpose(-2, -1)

        ## scale scores for numerical stability
        scores = scores * (self.head_size ** -0.5)

        ## apply the mask
        sequence_length = scores.shape[-1]
        mask = self.mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        ## masked self-attention
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        ## combine value vectors according to the attention weights
        return weights @ values

## extending the self-attention to multiple heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embed: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))

## feed forward network of the transformer model
class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.net(x))

class Block(nn.Module):
    def __init__(self, n_embed: int, num_heads: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if n_embed % num_heads != 0:
            raise ValueError("n_embed must be divisible by num_heads")
        head_size = n_embed // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout)
        self.feed_forward = FeedForward(n_embed, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## residual connection around the attention and feed forward layers
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embed: int, tokenizer: Tokenizer | None = None, num_heads: int = 2, n_layers: int = 1) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size ## the context length of the model
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(n_embed, num_heads, block_size, dropout=0.0)
                for _ in range(n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(n_embed)
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
        ## Transformer blocks
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Each position predicts the token at the corresponding position
            # in targets (the dataset has already shifted targets by one).
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return loss

        return logits

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size:]
            ## get the logits for the last position
            logits = self.forward(context)
            ## get the last position logits
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive")
                top_k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            ## sample the next token
            probs = F.softmax(logits, dim=-1)
            ## sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            ## append the next token to the tokens
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to generate text")
        token_ids = self.tokenizer.encode(prompt).ids
        device = next(self.parameters()).device
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = self.generate(
            tokens,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return self.tokenizer.decode(generated[0].tolist())

    ## At this point we have a model that is bigram model which means that the model predicts the next token based on only the previous token, but we want to take it to the next level and make it a transformer model.

    
