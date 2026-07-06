## distributed compute optimal training pipeline for any LLM

#### Build your own LLM or fine tune one with lowest possible cost

```bash
# install
uv pip install -e .

# pre-training (Gemma / TikToken)
uv run mint pretrain --config configs/config_d12_pretrain.toml

# supervised fine-tuning (Gemma / TikToken)
uv run mint sft --config configs/config_d12_sft.toml

# supervised fine-tuning (HuggingFace model)
uv run mint v2-sft \
    --config configs/config_hf_sft.toml \
    --model-name Qwen/Qwen2-0.5B \
    --datasets gsm8k smoltalk

uv run mint v2-sft \
    --config configs/config_hf_sft.toml \
    --model-name Qwen/Qwen2-0.5B \
    --datasets gsm8k smoltalk

# DPO (Gemma / TikToken)
uv run mint dpo --config configs/config_d12_dpo.toml --ref-model checkpoints/checkpoint_step_10.pt

# DPO (HuggingFace model)
uv run mint v2-dpo --model-name Qwen/Qwen2-0.5B --config configs/config_hf_dpo.toml

# data utilities
uv run mint prepare-dpo
uv run mint download-climbmix --help
# --train-args passes flags to the underlying train script as a single quoted string
uv run mint stream-train \
    --num-train-shards 1 \
    --min-shards 1 \
    --train-script scripts/pretrain.py \
    --train-args "--config configs/config_d12_pretrain.toml"

# list all commands
uv run mint --help
```

#### what's so special ?
- compute optimality: auto calculate all hyper params including dataset size, model size etc.. just based on FLOPS you have
- custom distributed dataloaders, evaluators
- distributed MuonAdamW with norMuon + polar express coeff
- ZeRO 2 sharding strategy
- all common sft datasets for MCQ, conversational, spellbee training
- python exec engine for on the fly math
- distributed pretraining, supervised fine tuning, direct preference optimization
- plug n play LoRA, QLoRA adapters
- MHA, GQA, MQA with FlashAttention
- RoPE, Sine-Cos, Linear Scaled RoPE, ALiBi
- KVCache
- experimentation: custom CUDA FlashAttention kernels 
