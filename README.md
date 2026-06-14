# Tiny GPT

PyTorch로 구현한 character-level Tiny GPT 프로젝트입니다. 이 프로젝트는 OpenAI GPT-2 전체를 복제하는 것이 아니라, 수업의 notebook_06 구조를 참고하여 Transformer 언어 모델의 핵심 원리를 학습하기 위해 작성되었습니다.

## 프로젝트 구조

- `data/raw.txt`: 원본 Gutenberg 텍스트
- `data/input.txt`: 정제된 입력 텍스트
- `src/__init__.py`
- `src/prepare_data.py`
- `src/model.py`
- `src/train.py`
- `src/generate.py`
- `checkpoints/.gitkeep`
- `outputs/sample.txt`
- `requirements.txt`
- `.gitignore`

## 실행 방법

```bash
python src/prepare_data.py
python src/train.py --max-steps 5 --eval-interval 5 --eval-iters 2
python src/generate.py --prompt "Elizabeth" --max-new-tokens 50
```

## 설명

이 모델은 character-level 토크나이저, token embedding, positional embedding, causal self-attention, Pre-LayerNorm Transformer block, 그리고 language modeling head를 포함하는 작은 GPT입니다.
