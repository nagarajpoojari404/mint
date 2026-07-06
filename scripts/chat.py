import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mint.sft.peft.lora import LoRA
from mint.utils.checkpointer import Checkpointer


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with any HuggingFace causal LM")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per turn (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable conversation history (single-turn mode)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trainer checkpoint (.pt) to load weights from",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Apply LoRA adapters before loading the checkpoint",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha / scaling numerator (default: 32)",
    )
    parser.add_argument(
        "--lora-targets",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Module name substrings to attach LoRA to (default: q_proj v_proj)",
    )
    args = parser.parse_args()

    print(f"Loading {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.lora:
        LoRA.apply(
            model=model,
            target_modules=args.lora_targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
        )
        model = model.to(model.device)

    if args.checkpoint is not None:
        print(f"Loading weights from checkpoint: {args.checkpoint}")
        Checkpointer.load_model(model, args.checkpoint)
        print("Checkpoint weights loaded.")

    model.eval()
    print("Ready. Type 'quit' or Ctrl-C to exit, 'reset' to clear history.\n")

    history: list[dict] = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            sys.exit(0)
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": args.system}]
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Build prompt via the tokenizer's chat template if available, else fall back
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            encoded = tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
        else:
            prompt = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history
                if m["role"] != "system"
            )
            if history[0]["role"] == "system":
                prompt = history[0]["content"] + "\n" + prompt
            prompt += "\nAssistant:"
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"Assistant: {reply}\n")

        if args.no_history:
            history = [{"role": "system", "content": args.system}]
        else:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()