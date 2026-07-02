import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mint.config.base import Config
from mint.data.dataloader import DataloaderConfig
from mint.data.datasets.arc import ARC
from mint.data.datasets.base import SFTEvalDataset
from mint.data.datasets.customjsonl import CustomJSON
from mint.data.datasets.gsm8k import GSM8K
from mint.data.datasets.humaneval import HumanEval
from mint.data.datasets.mmlu import MMLU
from mint.data.datasets.smoltalk import SmolTalk
from mint.data.datasets.spellingbee import SimpleSpelling, SpellingBee
from mint.data.dist.sft import DistributedSFTDataloader
from mint.optim.muon_adamw import MuonAdamWConfig
from mint.sft.peft.lora import LoRA
from mint.tokenizer import HFTokenizer
from mint.trainer.sft import HFSFTTrainer, SFTConfig
from mint.utils.device import Device, DeviceConfig
from mint.utils.logger import logger


@dataclass
class MetaConfig(Config):
    train: SFTConfig
    optim: MuonAdamWConfig
    device: DeviceConfig
    dl: DataloaderConfig


def parse_dataset_spec(spec: str, dataset_map: dict):  # noqa: ANN201
    if "*" in spec:
        count_str, name = spec.split("*", 1)
        count = int(count_str)
        name = name.strip()
    else:
        count = 1
        name = spec.strip()

    if name in dataset_map:
        return [dataset_map[name]() for _ in range(count)]

    try:
        logger.info(f"Loading CustomJSON dataset from {name} ({count} instance(s))")
        return [CustomJSON(filepath=name) for _ in range(count)]
    except Exception as e:
        logger.error(f"Failed to load dataset '{name}': {e}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for HuggingFace models")
    parser.add_argument(
        "--config", type=str, default="configs/config_hf_sft.toml", help="Path to config TOML file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen2-0.5B, microsoft/phi-2)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "gsm8k",
            "smoltalk",
            "arc",
            "mmlu",
            "humaneval",
            "spellingbee",
            "simplespelling",
        ],
        help="Datasets to train on (e.g., 'arc', '3*arc' for 3 instances, or a path to a .jsonl file)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True, help="Shuffle dataset mixture"
    )
    # ── LoRA args ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--lora", action="store_true", default=True,
        help="Apply LoRA adapters (default: True)",
    )
    parser.add_argument(
        "--no-lora", dest="lora", action="store_false",
        help="Disable LoRA and fine-tune all parameters",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha / scaling numerator (default: 32)",
    )
    parser.add_argument(
        "--lora-targets", type=str, nargs="+",
        default=["q_proj", "v_proj"],
        help="Module name substrings to attach LoRA to (default: q_proj v_proj)",
    )
    args = parser.parse_args()

    logger.info(f"Loading tokenizer: {args.model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = HFTokenizer(hf_tokenizer)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(
        f"BOS token: {tokenizer.bos_token}, "
        f"EOS token: {tokenizer.eos_token}, "
        f"PAD token: {tokenizer.pad_token}"
    )

    config: MetaConfig = MetaConfig.from_toml(toml_path=args.config)
    device = Device(config.device)

    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.use_amp else torch.float32,
        trust_remote_code=True,
    )

    # Resize embeddings when special tokens were added to the tokenizer
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    if tokenizer.vocab_size > original_vocab_size:
        logger.info(
            f"Resizing model embeddings from {original_vocab_size} to {tokenizer.vocab_size}"
        )
        model.resize_token_embeddings(tokenizer.vocab_size)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    if args.lora:
        LoRA.apply(
            model=model,
            target_modules=args.lora_targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"LoRA applied | rank={args.lora_r} | alpha={args.lora_alpha} | "
            f"targets={args.lora_targets} | trainable params={trainable:,}"
        )
    else:
        logger.info("LoRA disabled — fine-tuning all parameters")

    model = model.to(device.device)

    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    if config.train.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]
        logger.info("Compiled model")

    # Only pass trainable parameters to the optimizer (LoRA params when active,
    # all params otherwise).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.sched.matrix_lr if hasattr(config.train.sched, "matrix_lr") else 1e-5,
        weight_decay=(
            config.train.sched.weight_decay if hasattr(config.train.sched, "weight_decay") else 0.01
        ),
    )
    logger.info(
        f"Optimizer: AdamW | lr={optimizer.param_groups[0]['lr']} | "
        f"weight_decay={optimizer.param_groups[0]['weight_decay']}"
    )

    dataset_map = {
        "arc": lambda: ARC(subset="ARC-Challenge", split="train"),
        "mmlu": lambda: MMLU(subset="all", split="auxiliary_train"),
        "gsm8k": lambda: GSM8K(subset="main", split="train"),
        "humaneval": HumanEval,
        "smoltalk": lambda: SmolTalk(split="train"),
        "spellingbee": lambda: SpellingBee(size=1000, split="train"),
        "simplespelling": lambda: SimpleSpelling(size=1000, split="train"),
    }

    train_datasets = []
    for spec in args.datasets:
        train_datasets.extend(parse_dataset_spec(spec, dataset_map))

    eval_dataset_map = {
        "arc": lambda: ARC(subset="ARC-Challenge", split="test"),
        "mmlu": lambda: MMLU(subset="all", split="test"),
        "gsm8k": lambda: GSM8K(subset="main", split="test"),
        "humaneval": HumanEval,
        "spellingbee": lambda: SpellingBee(size=100, split="test"),
        "simplespelling": lambda: SimpleSpelling(size=100, split="test"),
    }

    unique_dataset_names = {type(ds).__name__.lower() for ds in train_datasets}
    eval_datasets = []
    for name in unique_dataset_names:
        if name in eval_dataset_map:
            ds = eval_dataset_map[name]()
            if isinstance(ds, SFTEvalDataset):
                eval_datasets.append(ds)

    logger.info(f"Training datasets: {args.datasets}")
    logger.info(f"Total training examples: {sum(len(ds) for ds in train_datasets):,}")
    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")

    dataloader = DistributedSFTDataloader(
        device=device,
        config=config.dl,
        tokenizer=tokenizer,
        datasets=train_datasets,
        shuffle=args.shuffle,
    )

    trainer = HFSFTTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        config=config.train,
        tokenizer=tokenizer,
        eval_datasets=eval_datasets,
    )

    trainer.train()


if __name__ == "__main__":
    main()
