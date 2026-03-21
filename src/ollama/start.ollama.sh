#!/bin/sh
set -e

echo ">>> Starting Ollama server in background..."
ollama serve &
sleep 4
echo ">>> Pulling model llama3.1..."
ollama pull llama3.1
echo ">>> Model ready!"
wait $!
