"""Entry-point dispatcher for the `mint` CLI.

Each sub-command maps to a script's existing ``main()`` function.
``sys.argv`` is rewritten so every script sees its own args unchanged —
no modifications to the original scripts are required.

Usage examples::

    uv run mint sft --help
    uv run mint dpo --model-name Qwen/Qwen2-0.5B
    uv run mint pretrain --config configs/config_d12_pretrain.toml
    uv run mint v2-sft --lora --datasets arc gsm8k
    uv run mint v2-dpo --config configs/config_hf_dpo.toml
    uv run mint download-climbmix --help
    uv run mint prepare-dpo --help
    uv run mint stream-train --help
    uv run mint chat --help
"""

import sys


# Maps CLI sub-command name → (importable module path, function name)
_COMMANDS: dict[str, tuple[str, str]] = {
    "sft": ("scripts.sfttrain", "main"),
    "dpo": ("scripts.dpo", "main"),
    "pretrain": ("scripts.pretrain", "main"),
    "v2-sft": ("scripts.v2.sft", "main"),
    "v2-dpo": ("scripts.v2.dpo", "main"),
    "download-climbmix": ("scripts.utils.async_download_climbmix", "main"),
    "prepare-dpo": ("scripts.utils.prepare_dpo_dataset", "main"),
    "stream-train": ("scripts.utils.train_with_streaming_download", "main"),
    "chat": ("scripts.chat", "main"),
    "exp-chat": ("scripts.exp.chat", "main"),
}


def _print_help() -> None:
    print("usage: mint <command> [args ...]\n")
    print("Available commands:")
    width = max(len(k) for k in _COMMANDS)
    descriptions = {
        "sft": "SFT training for Gemma model (TikToken tokenizer)",
        "dpo": "DPO training for Gemma model (TikToken tokenizer)",
        "pretrain": "Pre-training for Gemma model with distributed dataloader",
        "v2-sft": "SFT training for HuggingFace models (HF tokenizer)",
        "v2-dpo": "DPO training for HuggingFace models (HF tokenizer)",
        "download-climbmix": "Async download of the ClimbMix dataset",
        "prepare-dpo": "Prepare anthropic/hh-rlhf DPO dataset",
        "stream-train": "Train with streaming dataset download",
        "chat": "chat with model",
        "exp-chat": "ollama chat (experimental)",
    }
    for cmd, desc in descriptions.items():
        print(f"  {cmd:<{width}}  {desc}")
    print("\nRun `mint <command> --help` for command-specific options.")


def app() -> None:
    """Main entry point invoked by the `mint` console script."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        sys.exit(0)

    sub = sys.argv[1]
    if sub not in _COMMANDS:
        print(f"mint: unknown command '{sub}'\n", file=sys.stderr)
        _print_help()
        sys.exit(1)

    module_path, fn_name = _COMMANDS[sub]

    sys.argv = [f"mint {sub}", *sys.argv[2:]]

    import importlib

    mod = importlib.import_module(module_path)
    getattr(mod, fn_name)()
