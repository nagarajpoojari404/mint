import re
import json
from typing import Literal
from pydantic import BaseModel
from datasets import load_dataset

class ContentPart(BaseModel):
    type: Literal["text", "python", "python_output"]
    text: str

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

class Row(BaseModel):
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]


def parse_anthropic_chat(text: str) -> list[Message]:
    parts = re.split(r'\n\n(Human|Assistant):\s*', text)
    messages = []
    
    for i in range(1, len(parts), 2):
        role_str = parts[i]
        content_str = parts[i+1].strip() if i+1 < len(parts) else ""
        
        role = "user" if role_str == "Human" else "assistant"
        messages.append(Message(role=role, content=content_str))
        
    return messages


def main():
    print("Fetching anthropic/hh-rlhf from Hugging Face...")
    dataset = load_dataset("anthropic/hh-rlhf", split="train[:1000]")

    output_filename = "data/dpo/hh_rlhf_formatted.jsonl"
    print(f"Processing and writing rows to {output_filename}...")

    with open(output_filename, "w", encoding="utf-8") as f:
        for item in dataset:
            full_chosen_seq = parse_anthropic_chat(item["chosen"])
            full_rejected_seq = parse_anthropic_chat(item["rejected"])
            
            prompt_messages = full_chosen_seq[:-1]
            
            chosen_message = [full_chosen_seq[-1]]
            rejected_message = [full_rejected_seq[-1]]
            
            row = Row(
                prompt=prompt_messages,
                chosen=chosen_message,
                rejected=rejected_message
            )
            
            f.write(row.model_dump_json() + "\n")

    print(f"Successfully saved all formatted rows to {output_filename}!")

if __name__ == "__main__":
    main()