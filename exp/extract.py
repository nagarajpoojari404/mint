#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request


# --- CONFIGURATION ---
OLLAMA_MODEL = "gemma4:latest"  # Ensure you have pulled your preferred model
OLLAMA_URL = "http://localhost:11434/api/generate"
POLL_INTERVAL = 5
OUTPUT_JSONL = "training_data.jsonl"
TRACKING_FILE = ".processed_files.txt"

# Terminal Colors for clear debugging
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def query_ollama(prompt, step_name):
    """Talks to Ollama and logs detailed request/response payload debug info."""
    print(f"{BLUE}[DEBUG - {step_name}]{RESET} Sending prompt to Ollama ({len(prompt)} chars)...")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        },  # Slightly higher for creativity in multi-turn variation synthesis
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            elapsed = time.time() - start_time
            raw_response = json.loads(res_body).get("response", "").strip()

            print(f"{GREEN}[DEBUG - {step_name}]{RESET} Received response in {elapsed:.2f}s.")
            print(f"{MAGENTA}[RAW LLM RESPONSE]:{RESET}\n{raw_response}\n{MAGENTA}---{RESET}")
            return raw_response

    except urllib.error.URLError as e:
        print(f"{RED}[API ERROR - {step_name}]{RESET} Connection failed: {e.reason}")
        return ""


def clean_json_string(response_text):
    """Attempts to strip Markdown formatting code blocks if injected by the LLM."""
    cleaned = response_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip()
    return cleaned


def extract_comprehensive_context(conversation_text):
    """Step 1: Harvest explicit and implicit data points worth remembering."""
    prompt = f"""You are an advanced user-profiling and memory engine. Analyze this conversation history. Your job is to extract EVERYTHING worth remembering about the user's life, context, and profile.

Look for:
1. Explicit Facts: Name, job, tech stack, location, language, domain knowledge.
2. Implicit Preferences: Strong positive/negative impressions, implicit tastes (e.g., if they mention loving a vacation in Paris, remember they like Paris).
3. Professional/Personal Context: Ongoing projects, goals, roadblocks, tools they use (e.g., Ubuntu, Docker).
4. Social Landscape: Friends, family, coworkers, relationships mentioned (e.g., brother, boss).
5. Behavioral Nuances: Tone preference, humor style, or specific vocabulary quirks.

Output the results STRICTLY as a raw JSON array of strings. Do not write text before or after the JSON block.

Example Format:
[
  "The user runs Ubuntu (Linux) as a non-root user named johnwilliams.",
  "The user's brother is samwilliams, who possesses root privileges.",
  "The user shows a strong positive sentiment toward Docker but lacks experience configuring it.",
  "The user is working on a deployment pipeline project."
]

If no memorable profile/context information is found, output exactly: []

Conversation:
{conversation_text}

JSON Output:"""

    response = query_ollama(prompt, "STEP 1: COMPREHENSIVE EXTRACTION")
    cleaned = clean_json_string(response)

    try:
        facts = json.loads(cleaned)
        if isinstance(facts, list):
            return facts
        return []
    except json.JSONDecodeError:
        print(
            f"{YELLOW}[PARSING WARNING]{RESET} JSON parsing failed. Attempting newline split fallback."
        )
        return [
            line.strip("- *1234567890. ")
            for line in response.splitlines()
            if line.strip() and len(line) > 10
        ]


def generate_multi_turn_conversations(fact_context):
    """Step 2: Generate multi-turn synthetic chat transcripts utilizing the discovered context."""
    prompt = f"""You are a synthetic training data generator for fine-tuning personal assistant models.

You have been given a fact about the user:
FACT: "{fact_context}"

Your task: Generate exactly 4 completely distinct multi-turn conversations where the USER naturally asks questions that lead the assistant to apply or reveal this fact. 

Rules:
- The USER should ask questions as if they genuinely want to know something (e.g. "where do I live?", "what's my name?", "what OS am I on?"). Do NOT have the user state the fact themselves.
- The ASSISTANT should answer using the fact naturally and helpfully, as a personal assistant that knows the user well.
- Each conversation must have at minimum 2 user turns and 2 assistant turns.
- Conversations must feel natural — vary the tone (casual, curious, task-focused, etc.).
- Do NOT start every assistant reply with "Sure!" or "Of course!". Be direct.
- Output strictly as a valid JSON array of objects with a "messages" key.

Example (fact: "The user lives in New Delhi"):
[
  {{
    "messages": [
      {{"role": "user", "content": "Hey, where do I live again?"}},
      {{"role": "assistant", "content": "You live in New Delhi."}},
      {{"role": "user", "content": "Right, can you suggest some good weekend getaways from there?"}},
      {{"role": "assistant", "content": "Sure! From New Delhi, Rishikesh and Agra are great short trips — both under 5 hours by road."}}
    ]
  }}
]

Now generate 4 such conversations using the given FACT. JSON Output:"""

    response = query_ollama(prompt, "STEP 2: DIALOGUE SYNTHESIS")
    cleaned = clean_json_string(response)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        print(
            f"{RED}[PARSING ERROR]{RESET} Could not convert conversation synthesis into structured training JSON."
        )
        return []


def process_file(filepath):
    print(f"\n{YELLOW}=================================================={RESET}")
    print(f"📂 Found File: {filepath}")
    print(f"{YELLOW}=================================================={RESET}")

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print(f"{RED}[SKIP]{RESET} File structure doesn't match standard logs.")
            return False

        # Build comprehensive conversation layout
        convo_lines = []
        for msg in data:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                convo_lines.append(f"{msg['role'].upper()}: {msg['content']}")

        conversation_text = "\n".join(convo_lines)

        # Step 1: Deep Profile/Context Extraction
        contexts = extract_comprehensive_context(conversation_text)
        print(
            f"\n✨ {GREEN}[HARVESTING COMPLETE]{RESET} Discovered {len(contexts)} key contextual dimensions."
        )
        for ctx in contexts:
            print(f"  -> {ctx}")

        # Step 2: Multi-turn Training Data Amplification
        new_samples_count = 0
        if contexts:
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as jsonl_file:
                for ctx in contexts:
                    if not ctx:
                        continue
                    print(
                        f"\n🔄 Synthesizing conversational multi-turn variants for context: '{ctx}'"
                    )
                    conversations = generate_multi_turn_conversations(ctx)

                    for convo in conversations:
                        if isinstance(convo, dict) and "messages" in convo:
                            # Write each conversation directly as a single line in the JSONL
                            jsonl_file.write(json.dumps(convo, ensure_ascii=False) + "\n")
                            new_samples_count += 1

            print(
                f"\n💾 {GREEN}[SUCCESS]{RESET} Appended {new_samples_count} multi-turn conversation logs to '{OUTPUT_JSONL}'."
            )
        else:
            print(f"\nℹ️ {BLUE}[INFO]{RESET} No meaningful contexts extracted from this file.")

        return True
    except Exception as e:
        print(f"{RED}[CRITICAL ERROR]{RESET} Failed processing file {filepath}: {e}")
        return False


def main():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE) as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    print(f"{GREEN}=================================================={RESET}")
    print(f"🤖 Daemon Live | Harvester Model: {OLLAMA_MODEL}")
    print(f"📁 Tracking file history index size: {len(processed_files)}")
    print(f"⏳ Polling directory every {POLL_INTERVAL}s... Press Ctrl+C to stop.")
    print(f"{GREEN}=================================================={RESET}")

    while True:
        try:
            for file in os.listdir("."):
                if file.endswith(".json") and file != OUTPUT_JSONL and file not in processed_files:
                    if process_file(file):
                        with open(TRACKING_FILE, "a") as f:
                            f.write(f"{file}\n")
                        processed_files.add(file)
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[SHUTDOWN]{RESET} Daemon closed safely.")
            sys.exit(0)


if __name__ == "__main__":
    main()
