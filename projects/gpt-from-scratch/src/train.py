from pathlib import Path
import torch

from data import LanguageModelDataset
from model import GPT
from tokenizer import load_tokenizer

project_dir = Path(__file__).resolve().parent.parent
text = (project_dir / "data" / "sakespeare.txt").read_text(encoding="utf-8")
tokenizer = load_tokenizer(project_dir / "data" / "tokenizer.json")
dataset = LanguageModelDataset.from_text(text, tokenizer)

model = GPT(
    vocab_size=tokenizer.get_vocab_size(), 
    block_size=8,
    n_embed=8,
    tokenizer=tokenizer
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(10):
    inputs, targets = dataset.get_batch("train", batch_size=4, context_length=8)
    optimizer.zero_grad()
    loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch + 1}: loss = {loss.item():.4f}")


print(model.generate_text("Hello, I'm a language model.", 100))
