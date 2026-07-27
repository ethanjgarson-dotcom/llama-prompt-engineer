#!/usr/bin/env python3
"""
Hey Llama Daemon
Runs silently in the background, always listening for "hey llama".
When heard, opens a new Terminal window and launches llama_terminal.py.
"""
import subprocess
import time
import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE   = 16000
CHUNK_SECS    = 3
COOLDOWN_SECS = 5          # prevent double-triggering
WHISPER_MODEL = "base"

PYTHON  = "/Users/ethangarson/Desktop/ClaudeCodeTest/venv/bin/python"
SCRIPT  = "/Users/ethangarson/Desktop/LlamaProject/llama_terminal.py"

chunk_frames = SAMPLE_RATE * CHUNK_SECS


def open_llama_terminal():
    cmd = f"{PYTHON} {SCRIPT} --no-wake"
    # Open a new Terminal window and run the script inside it
    applescript = f'tell application "Terminal" to do script "{cmd}"'
    subprocess.run(["osascript", "-e", applescript])
    subprocess.run(["osascript", "-e", 'tell application "Terminal" to activate'])


def is_speech(audio, threshold=500):
    """Skip Whisper on silent chunks to save CPU."""
    return np.abs(audio).mean() > threshold


def main():
    print("Hey Llama daemon started. Listening for 'hey llama'...")
    model = whisper.load_model(WHISPER_MODEL)
    last_triggered = 0

    while True:
        audio = sd.rec(chunk_frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()

        if not is_speech(audio):
            continue

        audio_float = audio.astype(np.float32).flatten() / 32768.0
        try:
            text = model.transcribe(audio_float, language="en", fp16=False)["text"].strip().lower()
        except Exception:
            continue

        if "hey llama" in text or "hey, llama" in text:
            now = time.time()
            if now - last_triggered > COOLDOWN_SECS:
                last_triggered = now
                open_llama_terminal()


if __name__ == "__main__":
    main()
