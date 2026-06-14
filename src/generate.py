from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch
from torch import Tensor, nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from model import GPTConfig, TinyGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny GPT 텍스트 생성 스크립트")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--prompt", type=str, default="Elizabeth")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="outputs/sample.txt")
    return parser.parse_args()


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")
    return torch.load(path, map_location="cpu")


def decode_tokens(tokens: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[token] for token in tokens)


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config_data = checkpoint["model_config"]
    config = GPTConfig(
        vocab_size=config_data["vocab_size"],
        block_size=config_data["block_size"],
        emb_dim=config_data["emb_dim"],
        num_heads=config_data["num_heads"],
        num_layers=config_data["num_layers"],
        dropout=config_data["dropout"],
    )
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    itos = checkpoint["itos"]
    stoi = checkpoint["stoi"]

    if args.temperature <= 0:
        raise ValueError("temperature는 0보다 커야 합니다.")

    prompt_chars = list(args.prompt)
    invalid_chars = [ch for ch in prompt_chars if ch not in stoi]
    if invalid_chars:
        invalid_display = ", ".join(sorted(set(invalid_chars)))
        raise ValueError(f"알 수 없는 문자입니다: {invalid_display}")

    context = torch.tensor([stoi[ch] for ch in prompt_chars], dtype=torch.long).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    context = context.to(device)

    if context.size(1) > config.block_size:
        context = context[:, -config.block_size :]

    generated = context.tolist()[0]
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(context)
            logits = logits[:, -1, :] / max(args.temperature, 1e-8)
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())
            if context.size(1) > config.block_size:
                context = context[:, -config.block_size :]

    result = decode_tokens(generated, itos)
    print(result)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"생성 결과를 저장했습니다: {args.output}")


if __name__ == "__main__":
    main()
