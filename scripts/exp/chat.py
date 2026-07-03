#!/usr/bin/env python3
from pathlib import Path
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ollama_chat.py <model_name>")
        print("Example: python3 ollama_chat.py tiny-qwen")
        sys.exit(1)

    model_name = sys.argv[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("exp").mkdir(exist_ok=True)
    output_filename = f"exp/{timestamp}-{model_name}.json"
    
    messages = []

    print("=============================================")
    print(f" Starting chat with model: {model_name} (Streaming)")
    print(f" Saving session to: ./{output_filename}")
    print(" Type 'exit' or 'quit' to end the chat.")
    print("=============================================")
    print("")

    url = "http://localhost:11434/api/chat"

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting and saving chat history...")
            break

        if user_input.lower() in ['exit', 'quit'] or not user_input:
            print("Exiting and saving chat history...")
            break

        messages.append({"role": "user", "content": user_input})

        # Set "stream" to True
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }
        data = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            print("\nAI: ", end="", flush=True)
            assistant_response_chunks = []

            with urllib.request.urlopen(req) as response:
                # Read line by line as Ollama streams JSON objects per line
                for line in response:
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        # Safely extract text token from the chunk structure
                        content_piece = chunk.get('message', {}).get('content', '')
                        print(content_piece, end="", flush=True)
                        assistant_response_chunks.append(content_piece)
            
            print("\n") # Newline after the stream finishes
            
            full_response = "".join(assistant_response_chunks)
            messages.append({"role": "assistant", "content": full_response})
                
        except urllib.error.URLError as e:
            print(f"\n[Error connecting to Ollama]: {e.reason}")
            print("Is your Ollama app running?\n")
            messages.pop() # Remove the user message since the LLM didn't reply

    if messages:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"Chat history saved successfully to {output_filename}")
    else:
        print("No conversation history to save.")

if __name__ == "__main__":
    main()