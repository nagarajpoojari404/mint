#!/bin/bash
set -e

TARGET_DIR="./models"

HF_REPO=$1
GGUF_FILE=$2
OLLAMA_NAME=$3

if [ -z "$HF_REPO" ] || [ -z "$GGUF_FILE" ] || [ -z "$OLLAMA_NAME" ]; then
    echo "💡 Tip: You can also pass these arguments inline next time!"
    echo "   Example: $0 bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf llama3.2-3b"
    echo "--------------------------------------------------------"
    [ -z "$HF_REPO" ] && read -p "👤 Enter HF Repo ID (e.g., bartowski/Llama-3.2-3B-Instruct-GGUF): " HF_REPO
    [ -z "$GGUF_FILE" ] && read -p "📄 Enter exact GGUF Filename (e.g., Llama-3.2-3B-Instruct-Q4_K_M.gguf): " GGUF_FILE
    [ -z "$OLLAMA_NAME" ] && read -p "🤖 Enter name for Ollama (e.g., my-llama): " OLLAMA_NAME
fi

OLLAMA_NAME=$(echo "$OLLAMA_NAME" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

HF_DOWNLOAD_URL="https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILE}"
LOCAL_FILE_PATH="$TARGET_DIR/$GGUF_FILE"
MODELF_PATH="$TARGET_DIR/Modelfile.$OLLAMA_NAME"

mkdir -p "$TARGET_DIR"
if [ ! -f "$LOCAL_FILE_PATH" ]; then
    echo "📥 Downloading GGUF from Hugging Face..."
    curl -L "$HF_DOWNLOAD_URL" -o "$LOCAL_FILE_PATH"
else
    echo "✅ GGUF asset already cached locally."
fi

echo "📝 Creating configuration map..."
cat << EOF > "$MODELF_PATH"
FROM $LOCAL_FILE_PATH
PARAMETER temperature 0.7
EOF

# 4. Final Register & Execute
echo "🔨 Registering '$OLLAMA_NAME' directly with Ollama..."
ollama create "$OLLAMA_NAME" -f "$MODELF_PATH"

echo "🎉 Success! Initiating runtime environment..."
sleep 1
ollama run "$OLLAMA_NAME"