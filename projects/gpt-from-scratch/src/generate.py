"""Generate text from a trained GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT
from tokenizer import load_tokenizer


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT model")
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "checkpoints" / "gpt-tinystories.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=project_dir / "data" / "tokenizer.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    tokenizer = load_tokenizer(args.tokenizer)
    model = GPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embed=config["n_embed"],
        num_heads=config["num_heads"],
        n_layers=config["n_layers"],
        tokenizer=tokenizer,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(
        model.generate_text(
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
