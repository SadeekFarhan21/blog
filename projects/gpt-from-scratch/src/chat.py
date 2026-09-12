"""Interactive chat-style interface for a trained GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generate import select_device
from model import GPT
from tokenizer import load_tokenizer


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Chat with a trained GPT model")
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
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    history: list[tuple[str, str]] = []
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters on {device}.")
    print("Enter /clear to reset the conversation or /quit to exit.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in {"/quit", "/exit"}:
            break
        if user_message.lower() == "/clear":
            history.clear()
            print("Conversation cleared.\n")
            continue

        transcript = "".join(
            f"User: {user}\nAssistant: {assistant}\n"
            for user, assistant in history
        )
        prompt = f"{transcript}User: {user_message}\nAssistant:"
        prompt_ids = tokenizer.encode(prompt).ids
        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(
            tokens,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        response = tokenizer.decode(generated[0, len(prompt_ids):].tolist()).strip()
        response = response.split("\nUser:", 1)[0].strip()
        history.append((user_message, response))
        print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
