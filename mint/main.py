"""Entry-point dispatcher for the `mint` CLI.

Each sub-command maps to a script's ``main()`` function. ``sys.argv`` is rewritten
so every script sees its own args unchanged.

Usage examples::

    uv run mint sft --config configs/config_d12_sft.toml
    uv run mint dpo --ref-model checkpoints/checkpoint_step_10.pt
    uv run mint pretrain --config configs/config_d12_pretrain.toml
    uv run mint chat --model-name Qwen/Qwen2-0.5B-Instruct
"""

import importlib
import sys


# CLI sub-command -> (importable module path, description)
_COMMANDS: dict[str, tuple[str, str]] = {
    "sft": ("scripts.sfttrain", "SFT training for Gemma model (TikToken tokenizer)"),
    "dpo": ("scripts.dpo", "DPO training for Gemma model (TikToken tokenizer)"),
    "pretrain": ("scripts.pretrain", "Pre-training for Gemma model with distributed dataloader"),
    "chat": ("scripts.chat", "Chat with any HuggingFace causal LM"),
    "download-climbmix": ("scripts.utils.async_download_climbmix", "Async download of ClimbMix"),
    "download-eval": ("scripts.utils.download_eval_bundle", "Download evaluation benchmarks bundle"),
    "prepare-dpo": ("scripts.utils.prepare_dpo_dataset", "Prepare anthropic/hh-rlhf DPO dataset"),
    "stream-train": ("scripts.utils.train_with_streaming_download", "Train while downloading"),
    "gen-config": ("scripts.utils.gen_config_from_scaling_law", "Generate model config from scaling laws"),
}


def _print_help() -> None:
    print("usage: mint <command> [args ...]\n")
    print("Available commands:")
    width = max(map(len, _COMMANDS))
    for command, (_, description) in _COMMANDS.items():
        print(f"  {command:<{width}}  {description}")
    print("\nRun `mint <command> --help` for command-specific options.")


def app() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        sys.exit(0)

    command = sys.argv[1]
    if command not in _COMMANDS:
        print(f"mint: unknown command '{command}'\n", file=sys.stderr)
        _print_help()
        sys.exit(1)

    sys.argv = [f"mint {command}", *sys.argv[2:]]
    importlib.import_module(_COMMANDS[command][0]).main()


if __name__ == "__main__":
    app()
