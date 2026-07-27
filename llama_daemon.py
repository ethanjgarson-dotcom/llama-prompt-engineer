#!/usr/bin/env python3
"""
Llama Daemon — always-on background listener.
Runs silently at login. When it hears "hey llama" it opens a Terminal
window and launches llama_terminal.py.
"""
import time
import subprocess

import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE = 16000
CHUNK_SECS = 5       # longer window catches more context
SLIDE_SECS = 2       # overlap — re-checks every 2s so wake word is never split across chunks
COOLDOWN_SECS = 10

PYTHON     = "/Users/ethangarson/Desktop/ClaudeCodeTest/venv/bin/python"
WEB_SCRIPT = "/Users/ethangarson/Desktop/LlamaProject/llama_web.py"
WEB_URL    = "http://localhost:7337"

# Common ways Whisper mishears "hey llama"
WAKE_KEYWORDS = [
    "llama", "lama", "llamas", "lamas",
    "hey la", "a llama", "ay llama",
]


def is_wake_word(text):
    return any(kw in text for kw in WAKE_KEYWORDS)


_server_started = False

def open_llama_terminal():
    global _server_started
    if not _server_started:
        subprocess.Popen([PYTHON, WEB_SCRIPT],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _server_started = True
        time.sleep(1.5)
    subprocess.run(["open", WEB_URL])


def main():
    model = whisper.load_model("tiny")
    chunk_frames = SAMPLE_RATE * CHUNK_SECS
    slide_frames = SAMPLE_RATE * SLIDE_SECS
    buffer = np.zeros((chunk_frames, 1), dtype="int16")

    while True:
        new_audio = sd.rec(slide_frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()

        # Slide the buffer forward and append new audio
        buffer = np.concatenate([buffer[slide_frames:], new_audio], axis=0)
        audio_float = buffer.astype(np.float32).flatten() / 32768.0

        try:
            result = model.transcribe(
                audio_float,
                language="en",
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
            )
            text = result["text"].strip().lower()
            if text and is_wake_word(text):
                open_llama_terminal()
                buffer = np.zeros((chunk_frames, 1), dtype="int16")
                time.sleep(COOLDOWN_SECS)
        except Exception:
            pass


if __name__ == "__main__":
    main()
