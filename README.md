## nanochat like compute optimal training pipeline for Gemma models

#### Build your own LLM or fine tune one with lowest possible cost

```
uv run python scripts/download_eval_bundle.py

uv run python scripts/train_with_streaming_download.py --num-train-shards 1 --min-shards 1 --train-script pretrain.py --train-args="--config configs/config_d12_pretrain.toml"

uv run python sfttrain.py --config configs/config_d12_sft.toml

uv run scripts/prepare_dpo_dataset.py
uv run python dpo.py --config configs/config_d12_dpo.toml --ref-model checkpoints/checkpoint_step_10.pt

uv run python dpo_v2.py --model-name "Qwen/Qwen2-0.5B" --config configs/config_hf_dpo.toml 
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
