#!/bin/bash
# Llama launcher
# Usage:
#   ./start.sh          — voice mode terminal
#   ./start.sh --text   — text mode terminal
#   ./start.sh --web    — browser UI (recommended)
#   ./start.sh --daemon — background "hey llama" listener

VENV="/Users/ethangarson/Desktop/ClaudeCodeTest/venv/bin/python"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Make sure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 2
fi

if [ "$1" = "--daemon" ]; then
    echo "Starting Hey Llama daemon (background listener)..."
    "$VENV" "$DIR/llama_daemon.py"
elif [ "$1" = "--web" ]; then
    "$VENV" "$DIR/llama_web.py"
else
    "$VENV" "$DIR/llama_terminal.py" "$@"
fi
