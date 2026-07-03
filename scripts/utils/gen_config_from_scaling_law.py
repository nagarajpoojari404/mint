#!/usr/bin/env python3
import argparse
import math

import yaml


def calculate_hyperparams(
    depth_ref: int,
    depth_target: int,
    aspect_ratio: int,
    head_dim: int,
    seq_length: int,
    param_data_ratio: float,
    B_ref: int,  # noqa: N803
    lr_ref: float,
    wd_ref: float,
    vocab_size: int = 50258,
):

    # Calculate model dimensions
    base_dim = depth_target * aspect_ratio
    embed_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_heads = embed_dim // head_dim
    hidden_dim = 4 * embed_dim

    # Calculate scaling params (transformer matrices + lm_head)
    def get_scaling_params(depth):  # noqa: ANN202
        model_dim = depth * aspect_ratio
        model_dim = ((model_dim + head_dim - 1) // head_dim) * head_dim
        return 4 * depth * model_dim * model_dim

    N_ref = get_scaling_params(depth_ref)
    N_target = get_scaling_params(depth_target)

    # according to chinchilla scaling law. D ~ 20 * N
    # training horizon (tokens)
    D_ref = param_data_ratio * N_ref
    D_target = param_data_ratio * N_target

    # from power lines paper
    # batch size: B = B_ref × (D/D_ref)^0.383
    ratio_D = D_target / D_ref
    B_target = B_ref * (ratio_D**0.383)
    B_target = 2 ** round(math.log2(B_target))

    # from adamW theory
    # learning rate: η = η_ref × √(B/B_ref)
    ratio_B = B_target / B_ref
    lr_target = lr_ref * math.sqrt(ratio_B)

    # from T_epoch framework
    # weight decay: λ = λ_ref × √(B/B_ref) × (D_ref/D)
    wd_target = wd_ref * math.sqrt(ratio_B) * (D_ref / D_target)

    iters_target = int(D_target / B_target)

    # Calculate warmup/warmdown
    warmup_steps = max(100, int(0.02 * iters_target))
    warmdown_ratio = 0.65

    return {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "n_layers": depth_target,
        "n_heads": n_heads,
        "n_kv_heads": n_heads,
        "head_dim": head_dim,
        "batch_size": B_target,
        "train_num_steps": iters_target,
        "matrix_lr": lr_target,
        "weight_decay": wd_target,
        "warmup_steps": warmup_steps,
        "warmdown_ratio": warmdown_ratio,
        "scaling_params": N_target,
        "training_tokens": D_target,
    }


def generate_config_yaml(params, output_file) -> None:
    """Generate YAML config file from calculated parameters."""
    # Convert batch_size from tokens to sequences
    batch_size_sequences = params["batch_size"] // params["seq_length"]

    config = {
        "embed_dim": params["embed_dim"],
        "hidden_dim": params["hidden_dim"],
        "seq_length": params["seq_length"],
        "vocab_size": params["vocab_size"],
        "n_layers": params["n_layers"],
        "n_heads": params["n_heads"],
        "n_kv_heads": params["n_kv_heads"],
        "head_dim": params["head_dim"],
        "rope_base_theta": 100000,
        "tie_weights": True,
        "mixed_precision": True,
        "use_meta_device": True,
        "compile_model": True,
        "data_dir": "data/climbmix",
        "batch_size": batch_size_sequences,
        "buffer_size": 1000,
        "tokenizer_batch_size": 128,
        "train_num_steps": params["train_num_steps"],
        "grad_clip": 1.0,
        "log_every_n_steps": 1,
        "eval_every_n_steps": 100,
        "eval_num_steps": 10,
        "core_eval_every_n_step": 500,
        "gradient_accumulation_steps": 1,
        "checkpoint_dir": "checkpoints",
        "save_checkpoint_every_n_steps": 200,
        "keep_last_n_checkpoints": 1,
        "resume_from_checkpoint": None,
        "load_best_checkpoint": False,
        "wandb_enabled": True,
        "wandb_project": "goda",
        "wandb_run_name": None,
        "wandb_entity": None,
        "unembedding_lr": 0.004,
        "embedding_lr": 0.2,
        "matrix_lr": params["matrix_lr"],
        "scalar_lr": 0.5,
        "muon_momentum": 0.95,
        "muon_beta2": 0.9,
        "muon_ns_steps": 5,
        "adamw_beta1": 0.8,
        "adamw_beta2_lm_head": 0.96,
        "adamw_beta2_embedding": 0.995,
        "adamw_beta2_scalar": 0.95,
        "adamw_eps": 1.0e-10,
        "weight_decay": params["weight_decay"],
        "weight_decay_lm_head": 0.01,
        "weight_decay_embedding": 0.001,
        "weight_decay_scalar": 0.05,
        "scalar_lr_multiplier": 0.01,
        "warmup_steps": params["warmup_steps"],
        "warmdown_ratio": params["warmdown_ratio"],
        "final_lr_frac": 0.1,
        "muon_momentum_warmup_steps": 400,
        "muon_momentum_start": 0.85,
        "muon_momentum_peak": 0.97,
        "muon_momentum_final": 0.9,
    }

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def calculate_optimal_depth_from_flops(
    target_flops: float,
    aspect_ratio: int,
    head_dim: int,
    seq_length: int,
    param_data_ratio: float,
    vocab_size: int = 50258,
):
    # Chinchilla: N ∝ C^0.5, D ∝ C^0.5 (equal allocation)
    # where C is compute budget, N is params, D is data tokens

    # FLOPs per token ≈ 6N (forward + backward)
    # Total FLOPs = 6N × D
    # With D = ratio × N: FLOPs = 6N × ratio × N = 6 × ratio × N^2
    # So: N = sqrt(FLOPs / (6 × ratio))

    optimal_params = math.sqrt(target_flops / (6 * param_data_ratio))

    # binary search for depth that gives closest params
    # this is very rough approximation for params
    def get_params(depth):
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        # Non-embedding params only (for compute scaling)
        # Each layer: 4 × model_dim^2 (QKV + O + 2×MLP)
        return 4 * depth * model_dim * model_dim

    left, right = 1, 100
    best_depth = 12
    best_diff = float("inf")

    for depth in range(left, right + 1):
        params = get_params(depth)
        diff = abs(params - optimal_params)
        if diff < best_diff:
            best_diff = diff
            best_depth = depth

    return best_depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-ref", type=int, default=12)
    parser.add_argument("--depth-target", type=int, default=None)
    parser.add_argument("--flops", type=float, default=None, help="Total compute budget in FLOPs")
    parser.add_argument("--mfu", type=float, default=None, help="Model FLOPs Utilization (0-1)")
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--param-data-ratio", type=float, default=20)
    parser.add_argument("--B-ref", type=int, default=524288)
    parser.add_argument("--lr-ref", type=float, default=0.02)
    parser.add_argument("--wd-ref", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=50258)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Mode 1: Compute budget mode (--flops given)
    if args.flops is not None:
        if args.depth_target is not None:
            print("Warning: --depth-target ignored when --flops is specified")

        # Adjust for MFU if provided
        effective_flops = args.flops
        if args.mfu is not None:
            effective_flops = args.flops * args.mfu
            print(
                f"Adjusting for MFU {args.mfu:.2%}: {args.flops:.2e} → {effective_flops:.2e} FLOPs"
            )

        depth_target = calculate_optimal_depth_from_flops(
            effective_flops,
            args.aspect_ratio,
            args.head_dim,
            args.seq_length,
            args.param_data_ratio,
            args.vocab_size,
        )
        print(f"Compute-optimal depth for {args.flops:.2e} FLOPs: {depth_target}")

    # Mode 2: Direct depth specification (--depth-target given)
    elif args.depth_target is not None:
        depth_target = args.depth_target

    else:
        parser.error("Either --depth-target or --flops must be specified")

    result = calculate_hyperparams(
        args.depth_ref,
        depth_target,
        args.aspect_ratio,
        args.head_dim,
        args.seq_length,
        args.param_data_ratio,
        args.B_ref,
        args.lr_ref,
        args.wd_ref,
        args.vocab_size,
    )

    print(f"\nDepth: {result['n_layers']}")
    print(f"Embed dim: {result['embed_dim']}")
    print(f"Hidden dim: {result['hidden_dim']}")
    print(f"Num heads: {result['n_heads']}")
    print(f"Scaling params: {result['scaling_params']:,}")
    print(f"Training tokens: {result['training_tokens']:,}")
    print(f"Batch size (tokens): {result['batch_size']:,}")
    print(f"Batch size (sequences): {result['batch_size'] // args.seq_length}")
    print(f"Learning rate: {result['matrix_lr']:.6f}")
    print(f"Weight decay: {result['weight_decay']:.6f}")
    print(f"Iterations: {result['train_num_steps']:,}")
    print(f"Warmup steps: {result['warmup_steps']}")

    output_file = args.output or f"config_d{depth_target}.yaml"
    generate_config_yaml(result, output_file)
    print(f"\nConfig saved to: {output_file}")
