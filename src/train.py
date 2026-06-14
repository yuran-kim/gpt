from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from model import GPTConfig, NextTokenDataset, TinyGPT, sequence_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny GPT 학습 스크립트")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def split_data(data: Tensor) -> Tuple[Tensor, Tensor]:
    split = int(0.9 * data.size(0))
    return data[:split], data[split:]


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, eval_iters: int) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader):
            if step >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = sequence_cross_entropy(logits, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_text = load_data(os.path.join("data", "input.txt"))
    chars, stoi, itos = build_vocab(raw_text)
    data = encode(raw_text, stoi)
    if data.size(0) <= args.block_size:
        raise ValueError("데이터 길이가 block_size보다 짧습니다.")

    train_data, val_data = split_data(data)

    train_dataset = NextTokenDataset(train_data, args.block_size)
    val_dataset = NextTokenDataset(val_data, args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(
        vocab_size=len(chars),
        block_size=args.block_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = TinyGPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = 0
    while total_steps < args.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = sequence_cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            total_steps += 1
            if total_steps % args.eval_interval == 0 or total_steps == args.max_steps:
                val_loss = evaluate(model, val_loader, device, args.eval_iters)
                print(f"step {total_steps}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

            if total_steps >= args.max_steps:
                break

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "model_config": vars(config),
        "args": vars(args),
    }
    torch.save(checkpoint, os.path.join("checkpoints", "tiny_gpt.pt"))
    print("모델 체크포인트를 저장했습니다: checkpoints/tiny_gpt.pt")


if __name__ == "__main__":
    main()
