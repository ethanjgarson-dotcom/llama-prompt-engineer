#!/usr/bin/env python3
"""
Llama Daemon — voice activation + keyboard shortcut fallback.

Voice:    say "hey llama" to open Llama in the browser
Keyboard: press Cmd+Shift+Space as an instant fallback
"""
import time
import subprocess
import threading

import numpy as np
import sounddevice as sd
import whisper
from pynput import keyboard

SAMPLE_RATE     = 16000
FRAME_SECS      = 0.4
MAX_PHRASE_SECS = 2.5     # ignore phrases longer than this (mid-conversation speech)
SILENCE_FRAMES  = 3       # frames of silence before we cut and transcribe
COOLDOWN_SECS   = 8
CALIBRATE_SECS  = 2.0     # seconds to measure ambient noise at startup

PYTHON     = "/Users/ethangarson/Desktop/ClaudeCodeTest/venv/bin/python"
WEB_SCRIPT = "/Users/ethangarson/Desktop/LlamaProject/llama_web.py"
WEB_URL    = "http://localhost:7337"

WAKE_KEYWORDS = [
    "llama", "lama", "llamas", "lamas",
    "hey la", "a llama", "ay llama",
    "hiasma", "heyama", "hayama", "hey alma",
    "helama", "hey lama", "ey llama", "a lama",
    "hey, i'm up", "hey i'm up", "i'm up",
    "hey lamp", "hey lamps", "hey lam",
    "hela", "hella", "hey llama",
]


def is_wake_word(text):
    return any(kw in text for kw in WAKE_KEYWORDS)


_server_started = False
_cooldown       = False


def open_llama():
    global _server_started, _cooldown
    if _cooldown:
        return
    _cooldown = True

    if not _server_started:
        subprocess.Popen([PYTHON, WEB_SCRIPT],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _server_started = True
        time.sleep(2.0)

    subprocess.run(["open", WEB_URL])
    print("\n✓ Llama opened in browser.")

    def reset():
        global _cooldown
        time.sleep(COOLDOWN_SECS)
        _cooldown = False
        print("Listening for 'hey llama'…")

    threading.Thread(target=reset, daemon=True).start()


def calibrate():
    """Measure ambient noise so we can set a smart energy threshold."""
    print("Calibrating mic… stay quiet for 2 seconds.")
    frames = sd.rec(int(SAMPLE_RATE * CALIBRATE_SECS),
                    samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    ambient = np.sqrt(np.mean(frames.astype(np.float32) ** 2))
    # threshold = 4× ambient noise, but no lower than 60
    threshold = max(ambient * 4, 60)
    print(f"Ambient RMS: {ambient:.0f}  →  threshold: {threshold:.0f}")
    return threshold


def start_keyboard_listener():
    """Cmd+Shift+Space as instant fallback trigger."""
    def on_activate():
        print("Keyboard shortcut triggered.")
        open_llama()

    hotkey = keyboard.GlobalHotKeys({"<cmd>+<shift>+<space>": on_activate})
    hotkey.start()
    print("Keyboard shortcut: Cmd+Shift+Space")


def main():
    print("Loading Whisper…")
    model = whisper.load_model("base")  # base is more accurate than tiny

    threshold  = calibrate()
    frame_size = int(SAMPLE_RATE * FRAME_SECS)

    start_keyboard_listener()
    print("Listening for 'hey llama'…\n")

    speech_buf   = []
    silent_count = 0
    in_speech    = False

    while True:
        frame = sd.rec(frame_size, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()

        rms       = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        is_speech = rms > threshold

        if is_speech:
            speech_buf.append(frame)
            silent_count = 0
            in_speech    = True

        elif in_speech:
            speech_buf.append(frame)
            silent_count += 1

            if silent_count >= SILENCE_FRAMES:
                phrase_secs = len(speech_buf) * FRAME_SECS

                if phrase_secs <= MAX_PHRASE_SECS:
                    audio       = np.concatenate(speech_buf)
                    audio_float = audio.astype(np.float32).flatten() / 32768.0
                    try:
                        result = model.transcribe(
                            audio_float,
                            language="en",
                            fp16=False,
                            condition_on_previous_text=False,
                            no_speech_threshold=0.6,
                        )
                        text = result["text"].strip().lower()
                        if text:
                            print(f"Heard: {text}")
                        if text and is_wake_word(text):
                            open_llama()
                    except Exception as e:
                        print(f"Error: {e}")

                speech_buf   = []
                silent_count = 0
                in_speech    = False


if __name__ == "__main__":
    main()
