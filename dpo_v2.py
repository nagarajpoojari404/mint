import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mint.config.base import Config
from mint.data.dataloader import DataloaderConfig
from mint.data.dist.dpo import DistributedDPODataloader
from mint.optim.muon_adamw import MuonAdamWConfig
from mint.tokenizer import HFTokenizer
from mint.trainer.dpo import DPOConfig, DPOTrainer
from mint.utils.device import Device, DeviceConfig
from mint.utils.logger import logger


@dataclass
class MetaConfig(Config):
    train: DPOConfig
    optim: MuonAdamWConfig
    device: DeviceConfig
    dl: DataloaderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny open-source model with DPO")
    parser.add_argument(
        "--config", type=str, default="config_d12.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--min-shards",
        type=int,
        default=2,
        help="Minimum shards required before starting training",
    )
    parser.add_argument(
        "--max-shards-to-wait",
        type=int,
        default=-1,
        help="Maximum shards to wait for (-1 = wait indefinitely)",
    )
    parser.add_argument(
        "--ref-model",
        type=str,
        default=None,
        help="path to ref model checkpoint (optional, will use base model if not provided)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0, alternatives: Qwen/Qwen2-0.5B, microsoft/phi-2)",
    )
    args = parser.parse_args()

    # Load HuggingFace model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Wrap HuggingFace tokenizer with our HFTokenizer wrapper
    tokenizer = HFTokenizer(hf_tokenizer)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(
        f"BOS token: {tokenizer.bos_token}, EOS token: {tokenizer.eos_token}, PAD token: {tokenizer.pad_token}"
    )

    config: MetaConfig = MetaConfig.from_toml(toml_path=args.config)
    device = Device(config.device)

    # Load policy model
    logger.info("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.use_amp else torch.float32,
        trust_remote_code=True,
    )

    # Resize model embeddings to match tokenizer vocab size
    # This is critical when special tokens are added to the tokenizer
    original_vocab_size = policy_model.get_input_embeddings().weight.shape[0]
    if tokenizer.vocab_size > original_vocab_size:
        logger.info(
            f"Resizing model embeddings from {original_vocab_size} to {tokenizer.vocab_size}"
        )
        policy_model.resize_token_embeddings(tokenizer.vocab_size)

    # Load reference model (either from checkpoint or use the same base model)
    logger.info("Loading reference model...")
    if args.ref_model and args.ref_model != "-1":
        logger.info(f"Loading reference model from checkpoint: {args.ref_model}")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            torch_dtype=torch.bfloat16 if device.use_amp else torch.float32,
            trust_remote_code=True,
        )
    else:
        logger.info("Using base model as reference model")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if device.use_amp else torch.float32,
            trust_remote_code=True,
        )

    # Resize reference model embeddings as well
    ref_original_vocab_size = reference_model.get_input_embeddings().weight.shape[0]
    if tokenizer.vocab_size > ref_original_vocab_size:
        logger.info(
            f"Resizing reference model embeddings from {ref_original_vocab_size} to {tokenizer.vocab_size}"
        )
        reference_model.resize_token_embeddings(tokenizer.vocab_size)

    # Count parameters
    policy_params = sum(p.numel() for p in policy_model.parameters())
    ref_params = sum(p.numel() for p in reference_model.parameters())
    logger.info(f"Policy Model parameters: {policy_params:,}")
    logger.info(f"Reference Model parameters: {ref_params:,}")

    # Move models to device
    policy_model = policy_model.to(device.device)
    reference_model = reference_model.to(device.device)
    reference_model.requires_grad_(requires_grad=False)

    # Enable gradient checkpointing if configured
    if config.train.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for policy model")

    if config.train.compile_model:
        policy_model = torch.compile(policy_model)  # type: ignore[assignment]
        logger.info("Compiled policy model")

    # Configure optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.train.sched.matrix_lr if hasattr(config.train, "sched") else 1e-5,
        weight_decay=config.train.sched.weight_decay if hasattr(config.train, "sched") else 0.01,
    )
    logger.info(
        f"Optimizer configured with lr={optimizer.param_groups[0]['lr']}, weight_decay={optimizer.param_groups[0]['weight_decay']}"
    )

    dataloader: DistributedDPODataloader = DistributedDPODataloader(
        device=device,
        config=config.dl,
        tokenizer=tokenizer,
        filename="hh_rlhf_formatted.jsonl",
        min_shards_required=args.min_shards,
        max_shards_to_wait=args.max_shards_to_wait,
    )

    logger.info(f"Device: {device}")
    logger.info(f"Training on {device.type} with AMP={device.use_amp}")
    logger.info(f"Dataloader min_shards: {args.min_shards}")

    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        config=config.train,
        tokenizer=tokenizer,
    )

    # LoRA().apply(model=model, target_modules=[], r=8, alpha=16)  # noqa: ERA001
    trainer.train()


if __name__ == "__main__":
    main()
