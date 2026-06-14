from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """문자 기반 Next Token prediction을 위한 데이터셋.

    입력 시퀀스 x와 정답 시퀀스 y를 반환합니다.
    x shape: (T,), y shape: (T,)
    """

    def __init__(self, data: Tensor, block_size: int) -> None:
        if data.size(0) <= block_size:
            raise ValueError("데이터 길이가 block_size보다 짧습니다.")
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return self.data.size(0) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


class Head(nn.Module):
    """단일 self-attention head를 구현합니다.

    query, key, value를 별도의 bias 없는 Linear로 생성하고,
    causal mask를 사용해 미래 위치를 마스킹합니다.
    입력 shape: (B, T, emb_dim)
    출력 shape: (B, T, head_size)
    """

    def __init__(self, emb_dim: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # notebook_06과 동일하게 float tensor로 tril을 등록합니다.
        # 이후 마스킹 시 `== 0` 비교를 사용합니다.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(head_size)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # scaled dot-product attention (notebook_06 스타일 스케일링 사용)
        att = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        # 미래 토큰을 -inf로 마스킹 (tril이 0인 위치)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        return out


class MultiHeadAttention(nn.Module):
    """여러 attention head를 병렬로 계산하고 결과를 projection합니다.

    입력 shape: (B, T, emb_dim)
    출력 shape: (B, T, emb_dim)
    """

    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim은 num_heads로 나누어 떨어져야 합니다.")
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """MLP 블록: emb_dim -> 4*emb_dim -> emb_dim.

    입력 shape: (B, T, emb_dim)
    출력 shape: (B, T, emb_dim)
    """

    def __init__(self, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            # notebook_06에서는 ReLU를 사용합니다.
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer Block with Pre-LayerNorm.

    attention과 feed-forward sublayer에 residual 연결을 적용합니다.
    입력 shape: (B, T, emb_dim)
    출력 shape: (B, T, emb_dim)
    """

    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    emb_dim: int
    num_heads: int
    num_layers: int
    dropout: float


class TinyGPT(nn.Module):
    """작은 GPT 언어 모델.

    character-level token embedding, positional embedding, Transformer Block, 최종 LayerNorm,
    그리고 언어 모델 head를 포함합니다.
    입력 shape: (B, T)
    출력 shape: (B, T, vocab_size)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.emb_dim)
        self.blocks = nn.Sequential(
            *[
                Block(config.emb_dim, config.num_heads, config.block_size, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.head = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError("입력 길이 T가 block_size보다 큽니다.")
        token_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def sequence_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """시퀀스 예측을 위한 cross entropy loss를 계산합니다.

    logits shape: (B, T, vocab_size)
    targets shape: (B, T)
    출력 shape: scalar
    """
    # logits를 (B, C, T)로 변환하면 F.cross_entropy가 전체 시점에 대해
    # 올바르게 next-token loss를 계산합니다.
    loss = F.cross_entropy(logits.transpose(1, 2), targets)
    return loss
