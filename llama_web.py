#!/usr/bin/env python3
"""
Llama Web — browser-based UI for Llama Terminal
Runs a local Flask server and opens the browser automatically.
"""
import os
import sys
import json
import tempfile
import threading
import webbrowser
import time

import ollama
from flask import Flask, render_template, request, Response, jsonify, stream_with_context

app = Flask(__name__)

OLLAMA_MODEL     = "llama3.2:3b"
WHISPER_MODEL_ID = "base"
PORT             = 7337

_whisper      = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            import whisper as _w
            _whisper = _w.load_model(WHISPER_MODEL_ID)
    return _whisper


PROMPTS = {
    "refine":    "You are an expert prompt engineer. Transform the user's raw idea into a clear, specific, well-structured prompt. Preserve intent exactly. Output ONLY the refined prompt.",
    "precision": "You are an expert prompt engineer. Rewrite the idea as a numbered, step-by-step instruction set. Add a Constraints section. Output ONLY the precision prompt.",
    "creative":  "You are an expert prompt engineer for creative tasks. Rewrite as a vivid, evocative prompt. Use sensory, specific language. Output ONLY the creative prompt.",
    "code":      "You are an expert prompt engineer for software tasks. Rewrite as a precise technical specification: Task → Requirements → Constraints → Expected output. Output ONLY the spec.",
    "iterate":   "You have a refined prompt. Apply the user's requested adjustment precisely. Preserve everything not being changed. Output ONLY the adjusted prompt.",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        ollama.list()
        return jsonify({"ollama": True})
    except Exception:
        return jsonify({"ollama": False})


@app.route("/refine", methods=["POST"])
def refine():
    data    = request.json or {}
    raw     = data.get("text", "").strip()
    mode    = data.get("mode", "refine")
    current = data.get("current", "")

    if not raw:
        return jsonify({"error": "No text provided"}), 400

    system   = PROMPTS.get(mode, PROMPTS["refine"])
    user_msg = (
        f"Current prompt:\n\n{current}\n\nAdjustment: {raw}"
        if mode == "iterate" and current
        else f"Raw idea:\n\n{raw}"
    )

    def generate():
        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/improve", methods=["POST"])
def improve():
    data    = request.json or {}
    raw     = data.get("raw", "").strip()
    refined = data.get("refined", "").strip()

    if not raw or not refined:
        return jsonify({"error": "Missing inputs"}), 400

    system = (
        "You are an expert prompt engineer. Analyze the original idea and refined prompt, "
        "identify weak points, then generate exactly 3 improved alternative prompts.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{"analysis": "1-2 sentence analysis of weak points", '
        '"alternatives": ["option 1", "option 2", "option 3"]}'
    )
    user_msg = f"Original idea:\n{raw}\n\nRefined prompt:\n{refined}"

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            format="json",
        )
        text   = response["message"]["content"]
        result = json.loads(text)
        # Ensure required keys exist
        result.setdefault("analysis", "")
        result.setdefault("alternatives", [])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio"}), 400

    audio_file = request.files["audio"]
    model      = get_whisper()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="en", fp16=False)
        text   = result["text"].strip()
    except Exception:
        text = ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return jsonify({"text": text})


if __name__ == "__main__":
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"  Llama Web running at http://localhost:{PORT}")
    app.run(host="localhost", port=PORT, threaded=True, debug=False)
