# mint

**A compute-optimal, distributed training pipeline for LLMs.**
Pre-train, fine-tune and align a model from scratch on the FLOPs you actually have.

mint ships a Gemma-style decoder, a custom distributed data/eval stack, a distributed
Muon + AdamW optimizer with ZeRO sharding and tensor parallelism, and one CLI that drives
pre-training, SFT and DPO.

---

## Highlights

- **Compute-optimal by default**: derive model size, token budget, batch size, LR and
  weight decay from a FLOPs budget using scaling laws (`mint gen-config`).
- **Multi-GPU and multi-node training**: scale from one GPU to many nodes with `torchrun`
  (NCCL on CUDA, Gloo otherwise).
- **Tensor parallelism (TP)**: column / row-parallel linear layers driven by a per-model
  TP plan, with attention heads split across the TP group.
- **Sequence parallelism (SP)**: shards activations along the sequence dimension on top of
  TP, using all-gather / reduce-scatter in place of all-reduce.
- **Data parallelism with ZeRO-2 / ZeRO-3 sharding**, composed with TP on a 2-D
  `dp` x `tp` `DeviceMesh`.
- **Distributed MuonAdamW**: NorMuon with Polar Express coefficients for matrix
  parameters, AdamW for embeddings, LM head and scalars.
- **Full post-training stack**: pre-training, supervised fine-tuning and direct
  preference optimization (DPO).
- **Adapters**: plug-and-play LoRA and QLoRA.
- **Modern attention**: MHA, GQA and MQA with FlashAttention, plus a KV cache for inference.
- **Positional encodings**: RoPE, linearly scaled RoPE, sinusoidal, ALiBi and NoPE.
- **Batteries-included data**: streaming ClimbMix pre-training data, and SFT datasets for
  MCQ, math, code, conversation and spelling (see [Datasets](#datasets)).
- **Sandboxed Python execution engine** for tool-style math and code evaluation.
- **Custom CUDA FlashAttention kernels** (GQA / MQA) in [`kernels_/`](kernels_).
- **Tooling**: W&B logging, `torch.profiler` integration, checkpoint rotation and resume.

## Installation

Requires Python 3.12 to 3.14 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is
recommended for real training runs.

```bash
git clone https://github.com/nagarajpoojari404/mint.git
cd mint
uv sync            # or: uv pip install -e .
uv run mint --help
```

## Quick start

All functionality is exposed through the `mint` CLI. Each sub-command forwards its
arguments to the matching script in [`scripts/`](scripts).

```bash
# 1. Download pre-training data (ClimbMix shards)
uv run mint download-climbmix --help

# 2. Pre-train
uv run mint pretrain --config configs/config_d12_pretrain.toml

# 3. Supervised fine-tuning
uv run mint sft --config configs/config_d12_sft.toml

# 4. Preference optimization (DPO)
uv run mint prepare-dpo
uv run mint dpo --config configs/config_d12_dpo.toml \
    --ref-model checkpoints/checkpoint_step_10.pt

# 5. Chat with a model
uv run mint chat --model-name Qwen/Qwen2-0.5B-Instruct
```

### Commands

| Command             | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `pretrain`          | Pre-train the Gemma model with the distributed dataloader |
| `sft`               | Supervised fine-tuning (TikToken tokenizer)              |
| `dpo`               | Direct preference optimization (TikToken tokenizer)      |
| `chat`              | Chat with any HuggingFace causal LM                      |
| `gen-config`        | Generate a model config from scaling laws                |
| `download-climbmix` | Async download of the ClimbMix pre-training dataset      |
| `download-eval`     | Download the evaluation benchmark bundle                 |
| `prepare-dpo`       | Prepare the `anthropic/hh-rlhf` DPO dataset              |
| `stream-train`      | Train while shards are still downloading                 |

Run `uv run mint <command> --help` for the options of any command.

### Train while you download

`stream-train` starts training once a minimum number of shards is available, so you do
not wait for the full dataset. `--train-args` passes flags to the training script as a
single quoted string.

```bash
uv run mint stream-train \
    --num-train-shards 1 \
    --min-shards 1 \
    --train-script scripts/pretrain.py \
    --train-args "--config configs/config_d12_pretrain.toml"

# multi-GPU
uv run mint stream-train ... --use-torchrun --torchrun-args "--nproc_per_node=2"
```

### Distributed training (multi-GPU, multi-node, TP, SP)

Launch with `torchrun`, and choose the parallelism layout in the `[device]` table of your config:

```toml
[device]
mixed_precision = true
tp_size   = 2      # tensor-parallel group size (world_size must be divisible by it)
ddp_level = 2      # 0/1: plain data parallel, 2: ZeRO-2, 3: ZeRO-3
sp        = true   # sequence parallel, shards activations across the TP group
```

`world_size = dp_size x tp_size`. Ranks are laid out so that consecutive ranks share a TP
group, which keeps tensor-parallel traffic on the fastest links (ideally within one node).
With `world_size = 4` and `tp_size = 2`, mint builds a 2 x 2 (`dp` x `tp`) device mesh.

**Single node, multiple GPUs**

```bash
torchrun --nproc_per_node=4 -m scripts.pretrain --config configs/config_d12_pretrain.toml
```

**Multiple nodes** (run on every node, with `--node_rank` set to 0, 1, ...)

```bash
torchrun \
    --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<node-0-ip> --master_port=29500 \
    -m scripts.pretrain --config configs/config_d12_pretrain.toml
```

Or use rendezvous instead of a fixed master:
`--rdzv_backend=c10d --rdzv_endpoint=<node-0-ip>:29500`.

| Setting     | Requirement                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| `tp_size`   | Must divide `world_size`; attention heads and KV heads must divide by it     |
| `sp = true` | Needs `tp_size > 1`; sequence length must divide by `tp_size`                |
| `ddp_level` | `3` shards parameters as well; the data-parallel group size is `world_size / tp_size` |

The same launch works for `sft` and `dpo`, and `stream-train` accepts
`--use-torchrun --torchrun-args "..."`.

## Compute-optimal configs

Instead of hand-tuning hyper-parameters, give mint a compute budget and let it work out
the rest: model depth and width, tokens to train on (Chinchilla-style `D ~ 20 * N`),
batch size, learning rates and weight decay.

```bash
uv run mint gen-config --help
```

Key options: `--flops` (total compute budget), `--mfu` (expected model FLOPs
utilization), `--depth-target`, `--aspect-ratio`, `--head-dim`, `--seq-length`,
`--param-data-ratio` and `--output`.

## Configuration

Runs are described by TOML files in [`configs/`](configs). A config has these sections:

| Section            | Controls                                                          |
| ------------------ | ----------------------------------------------------------------- |
| `[model]`          | Embedding / hidden size, layers, heads, KV heads, RoPE, tied weights |
| `[train]`          | Steps, grad clipping, accumulation, compile, mixed precision      |
| `[train.sched]`    | Per-group learning rates, warmup / warmdown, Muon momentum, decay |
| `[train.ckpt]`     | Checkpoint directory, frequency, rotation, resume                 |
| `[train.lg]`       | Weights & Biases logging                                          |
| `[train.profiler]` | `torch.profiler` schedule and outputs                             |
| `[optim]`          | Muon and AdamW hyper-parameters                                   |
| `[device]`         | Mixed precision and parallelism (`tp_size`, `ddp_level`, `sp`)    |
| `[dl]`             | Data directory, batch size, sequence length, buffers              |

Available configs: `config_d12_pretrain.toml`, `config_d12_dpo.toml`,
`config_hf_sft.toml`, `config_hf_dpo.toml`.

## Datasets

SFT mixes are chosen with `--datasets`. Prefix a name with `N*` to repeat it N times,
or pass a path to a custom `.jsonl` file.

```bash
uv run mint sft --config configs/config_d12_sft.toml \
    --datasets gsm8k smoltalk 3*arc path/to/custom.jsonl
```

| Name             | Task                    |
| ---------------- | ----------------------- |
| `gsm8k`          | Grade-school math       |
| `smoltalk`       | Conversational          |
| `arc`            | Science MCQ             |
| `mmlu`           | Multi-task MCQ          |
| `humaneval`      | Code generation         |
| `spellingbee`    | Spelling / letter tasks |
| `simplespelling` | Simple spelling         |

## Project layout

```
mint/
├── mint/                 # library
│   ├── nn/               # attention, FlashAttention, RoPE/ALiBi, FFN, norms, blocks
│   ├── models/           # Gemma
│   ├── optim/            # MuonAdamW and its distributed variant
│   ├── trainer/          # pretrain / SFT / DPO trainers and LR scheduler
│   ├── sft/peft/         # LoRA and QLoRA
│   ├── loss/             # DPO loss
│   ├── data/             # datasets, samplers, distributed dataloaders
│   ├── eval/             # bits-per-byte, CORE and chat evaluators
│   ├── kvcache/          # KV cache for generation
│   ├── tokenizer/        # TikToken tokenizer
│   ├── utils/            # parallelism (TP / ZeRO), device, checkpointing, profiler, exec
│   └── main.py           # `mint` CLI dispatcher
├── scripts/              # entry points behind each CLI command
├── configs/              # TOML run configs
├── kernels_/             # experimental CUDA FlashAttention kernels (GQA, MQA)
└── tests/                # unit and distributed integration tests
```

## Development

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

Tests under [`tests/unit`](tests/unit) cover ZeRO sharding and tensor parallelism;
[`tests/integration`](tests/integration) exercises distributed training.

## License

No license has been specified yet.
