import argparse
from pathlib import Path
import torch

from data import LanguageModelDataset, tokenize_file
from model import GPT
from tokenizer import load_tokenizer

def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train the small GPT model")
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "tinystories.txt")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="maximum characters to load; default tokenizes and caches the entire corpus",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=project_dir / "checkpoints" / "gpt-tinystories.pt")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.embed_size % args.heads != 0:
        parser.error("--embed-size must be divisible by --heads")
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive")

    torch.manual_seed(args.seed)

    tokenizer_path = project_dir / "data" / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    if args.max_chars is None:
        token_cache = args.data.with_suffix(args.data.suffix + ".tokens.uint16")
        cache_is_stale = not token_cache.exists() or token_cache.stat().st_mtime_ns < max(
            args.data.stat().st_mtime_ns,
            tokenizer_path.stat().st_mtime_ns,
        )
        if cache_is_stale:
            token_count = tokenize_file(args.data, token_cache, tokenizer)
            print(f"cached {token_count:,} tokens at {token_cache}")
        dataset = LanguageModelDataset.from_token_file(token_cache, tokenizer)
    else:
        with args.data.open(encoding="utf-8") as data_file:
            text = data_file.read(args.max_chars)
        dataset = LanguageModelDataset.from_text(text, tokenizer)
    model = GPT(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=args.block_size,
        n_embed=args.embed_size,
        num_heads=args.heads,
        n_layers=args.layers,
        tokenizer=tokenizer,
    )
    print(f"data={args.data.name} parameters={sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for step in range(args.steps):
        inputs, targets = dataset.get_batch("train", args.batch_size, args.block_size)
        optimizer.zero_grad()
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            print(f"step {step + 1}/{args.steps}: loss = {loss.item():.4f}")

    print(model.generate_text("To be", 50))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": tokenizer.get_vocab_size(),
            "block_size": args.block_size,
            "n_embed": args.embed_size,
            "num_heads": args.heads,
            "n_layers": args.layers,
        },
        "training": vars(args),
        "final_loss": loss.item(),
        "tokenizer": str((project_dir / "data" / "tokenizer.json").resolve()),
    }, args.output)
    print(f"saved checkpoint={args.output} size={args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
