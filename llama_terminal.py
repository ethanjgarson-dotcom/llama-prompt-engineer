#!/usr/bin/env python3
"""
Llama Terminal — v3
Near-monochrome UI · streaming output · smooth animations
"""
import sys
import os
import termios
import tty
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import sounddevice as sd
import whisper
import ollama
import pyperclip

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.spinner import Spinner
from rich.rule import Rule
from rich.columns import Columns

# ── Console ───────────────────────────────────────────────────────────────────
console = Console()

# ── Palette ───────────────────────────────────────────────────────────────────
TEXT       = "grey93"       # warm off-white (terminal safe)
TEXT_DIM   = "grey50"       # muted gray
TEXT_FAINT = "grey23"       # barely visible
BORDER     = "grey19"       # dark border
BTN_BG     = "white"        # primary button background
BTN_FG     = "black"        # primary button text

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
WHISPER_MODEL = "base"
OLLAMA_MODEL  = "llama3.2:3b"

# ── Session state ─────────────────────────────────────────────────────────────
_history   = []
_text_mode = False

# ── Modes ─────────────────────────────────────────────────────────────────────
MODES = {
    "1": ("refine",    "Quick Refine", "Clean, structured prompt from your raw idea"),
    "2": ("precision", "Precision",    "Step-by-step, zero-ambiguity instructions"),
    "3": ("creative",  "Creative",     "Vivid, evocative prompt for creative tasks"),
    "4": ("code",      "Code",         "Technical spec for engineering tasks"),
}

# ── System prompts ────────────────────────────────────────────────────────────
PROMPTS = {
    "refine": """\
You are an expert prompt engineer. Transform the user's raw idea into a clear, specific,
well-structured prompt. Preserve intent exactly. Output ONLY the refined prompt.""",

    "precision": """\
You are an expert prompt engineer. Rewrite the idea as a numbered, step-by-step instruction
set. Add a Constraints section. Output ONLY the precision prompt.""",

    "creative": """\
You are an expert prompt engineer for creative tasks. Rewrite as a vivid, evocative prompt.
Use sensory, specific language. Output ONLY the creative prompt.""",

    "code": """\
You are an expert prompt engineer for software tasks. Rewrite as a precise technical
specification: Task → Requirements → Constraints → Expected output. Output ONLY the spec.""",

    "iterate": """\
You have a refined prompt. Apply the user's requested adjustment precisely.
Preserve everything not being changed. Output ONLY the adjusted prompt.""",
}


# ── Utilities ─────────────────────────────────────────────────────────────────
def flush_stdin():
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass



def clear():
    os.system("clear")


# ── UI builders ───────────────────────────────────────────────────────────────
def build_header(active_mode_id=None):
    """Top bar — logo · mode tabs · toggle."""
    tabs = []
    for key, (mode_id, label, desc) in MODES.items():
        if mode_id == active_mode_id:
            tabs.append(f"[bold {TEXT}]{label}[/bold {TEXT}]")
        else:
            tabs.append(f"[{TEXT_FAINT}]{label}[/{TEXT_FAINT}]")

    mode_str = "Text" if _text_mode else "Voice"

    grid = Table.grid(expand=True, padding=(0, 0))
    grid.add_column(ratio=1)
    grid.add_column(ratio=3, justify="center")
    grid.add_column(ratio=1, justify="right")
    grid.add_row(
        f"[bold {TEXT}]● Llama[/bold {TEXT}]  [dim]v3[/dim]",
        "  ".join(tabs),
        f"[{TEXT_DIM}]{mode_str}[/{TEXT_DIM}]  [{TEXT_FAINT}]History[/{TEXT_FAINT}]",
    )
    return Panel(grid, border_style=BORDER, padding=(0, 2))


def build_left(raw=None, active_mode_id=None):
    """Left panel — input preview + mode list + refine button."""
    rows = Table.grid(padding=(0, 0))
    rows.add_column(min_width=28)

    # Input label
    rows.add_row(Text("Your idea", style=TEXT_DIM))
    rows.add_row(Text(""))

    # Input preview box
    if raw:
        inner = Text(raw, style=TEXT, overflow="fold")
        footer = Text(f"{len(raw)} chars", style=TEXT_FAINT)
    else:
        inner  = Text("Enter your idea below…", style=TEXT_FAINT)
        footer = Text("")

    input_grid = Table.grid(padding=(0, 0))
    input_grid.add_column()
    input_grid.add_row(inner)
    input_grid.add_row(Text(""))
    input_grid.add_row(footer)
    rows.add_row(Panel(input_grid, border_style=BORDER, padding=(0, 1)))

    rows.add_row(Text(""))
    rows.add_row(Rule(style=BORDER))
    rows.add_row(Text(""))
    rows.add_row(Text("Mode", style=TEXT_DIM))
    rows.add_row(Text(""))

    # Mode list
    for key, (mode_id, label, desc) in MODES.items():
        active = (mode_id == active_mode_id)
        name_style = f"bold {TEXT}" if active else TEXT_FAINT
        dot = "●" if active else "·"
        rows.add_row(Text(f"{dot}  {label}", style=name_style))
        if active:
            rows.add_row(Text(f"   {desc}", style=TEXT_DIM))
        rows.add_row(Text(""))

    rows.add_row(Text(""))

    # Primary button — white background, dark text
    btn = Table.grid(expand=True)
    btn.add_column(justify="center")
    btn.add_row(Text("Refine   ⌘ ↵", style=f"bold {BTN_FG} on {BTN_BG}"))
    rows.add_row(Panel(btn, border_style=BTN_BG, padding=(0, 2), height=3))

    return Panel(rows, border_style=BORDER, padding=(0, 1))


def build_right_empty():
    """Right panel — empty/waiting state."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    grid.add_row(Text(""))
    grid.add_row(Text("Your refined prompt will appear here.", style=TEXT_FAINT))
    grid.add_row(Text(""))
    return Panel(Align.center(grid, vertical="middle"), border_style=BORDER, padding=(0, 2))


def build_right_loading(elapsed=0.0):
    """Right panel — animated loading state."""
    frames = ["●  ○  ○", "●  ●  ○", "●  ●  ●", "○  ●  ●", "○  ○  ●", "○  ○  ○"]
    dot    = frames[int(elapsed * 3) % len(frames)]

    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    grid.add_row(Text(""))
    grid.add_row(Text(dot, style=TEXT_DIM))
    grid.add_row(Text(""))
    grid.add_row(Text("Refining your prompt", style=TEXT_FAINT))
    grid.add_row(Text(""))
    grid.add_row(Text(""))

    # Skeleton lines
    for w in [44, 32, 40, 26, 38, 20]:
        grid.add_row(Text("─" * w, style=TEXT_FAINT))
        grid.add_row(Text(""))

    return Panel(Align.center(grid, vertical="middle"), border_style=BORDER, padding=(0, 2))


def build_right_output(output, mode_id, copied=False):
    """Right panel — result display."""
    mode_label = next((v[1] for v in MODES.values() if v[0] == mode_id), "Output")
    copy_text  = f"[{TEXT_DIM}]✓ Copied[/{TEXT_DIM}]" if copied else f"[{TEXT_FAINT}]c  copy[/{TEXT_FAINT}]"

    body = Table.grid(padding=(0, 0))
    body.add_column()

    # Header row
    hdr = Table.grid(expand=True)
    hdr.add_column(ratio=1)
    hdr.add_column(ratio=1, justify="right")
    hdr.add_row(Text(mode_label, style=TEXT_FAINT), Text.from_markup(copy_text))
    body.add_row(hdr)
    body.add_row(Text(""))

    # Output text
    body.add_row(Text(output, style=TEXT, overflow="fold"))

    return Panel(body, border_style=BORDER, padding=(0, 2))


def render_screen(raw=None, active_mode_id=None, output=None, copied=False):
    """Print the full UI: header + two panels."""
    clear()
    console.print(build_header(active_mode_id))
    left  = build_left(raw, active_mode_id)
    right = build_right_output(output, active_mode_id, copied) if output else build_right_empty()
    console.print(Columns([left, right], expand=True))


# ── LLM — streaming ───────────────────────────────────────────────────────────
def stream_llm(system_prompt, user_content):
    """Yield progressively longer result strings as tokens arrive."""
    accumulated = ""
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        accumulated += token
        yield accumulated


def run_mode(mode_id, raw_text):
    """Run a mode with animated loading then streaming output."""
    system = PROMPTS[mode_id]
    user_msg = f"Raw idea:\n\n{raw_text}"

    left   = build_left(raw_text, mode_id)
    header = build_header(mode_id)
    result = ""
    start  = time.time()

    def make_live_layout(right_panel):
        layout = Layout()
        layout.split_column(
            Layout(header,       name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(left,         name="left",  ratio=2),
            Layout(right_panel,  name="right", ratio=3),
        )
        return layout

    with Live(
        make_live_layout(build_right_loading(0)),
        console=console,
        refresh_per_second=12,
        screen=True,
    ) as live:
        # Hold loading animation briefly
        while time.time() - start < 0.8:
            live.update(make_live_layout(build_right_loading(time.time() - start)))
            time.sleep(0.08)

        # Stream result in
        for partial in stream_llm(system, user_msg):
            result = partial
            live.update(make_live_layout(build_right_output(result, mode_id)))

    return result.strip()


def run_iterate(current_result, adjustment, mode_id):
    """Stream an adjusted result."""
    user_msg = f"Current prompt:\n\n{current_result}\n\nAdjustment: {adjustment}"
    left   = build_left(adjustment, mode_id)
    header = build_header(mode_id)
    result = ""
    start  = time.time()

    def make_live_layout(right_panel):
        layout = Layout()
        layout.split_column(
            Layout(header,      name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(left,        name="left",  ratio=2),
            Layout(right_panel, name="right", ratio=3),
        )
        return layout

    with Live(
        make_live_layout(build_right_loading(0)),
        console=console,
        refresh_per_second=12,
        screen=True,
    ) as live:
        while time.time() - start < 0.6:
            live.update(make_live_layout(build_right_loading(time.time() - start)))
            time.sleep(0.08)
        for partial in stream_llm(PROMPTS["iterate"], user_msg):
            result = partial
            live.update(make_live_layout(build_right_output(result, mode_id)))

    return result.strip()


def copy_to_clipboard(text):
    try:
        pyperclip.copy(text)
        return True
    except Exception:
        return False


# ── Commands ──────────────────────────────────────────────────────────────────
def handle_command(cmd):
    c = cmd.strip().lower()
    if c == "/history":
        show_history(); return True
    if c == "/help":
        show_help();    return True
    if c == "/clear":
        clear();        return True
    if c in ("/quit", "/q", "/exit"):
        console.print(f"\n[{TEXT_DIM}]Goodbye.[/{TEXT_DIM}]\n")
        sys.exit(0)
    return False


def show_history():
    clear()
    if not _history:
        console.print(f"[{TEXT_DIM}]  No history yet.[/{TEXT_DIM}]\n")
        return
    t = Table(border_style=BORDER, header_style=f"bold {TEXT_DIM}")
    t.add_column("#",      style=TEXT_FAINT, width=3)
    t.add_column("Mode",   style=TEXT_DIM,   width=14)
    t.add_column("Time",   style=TEXT_FAINT, width=6)
    t.add_column("Input",  style=TEXT_DIM,   max_width=36)
    t.add_column("Output", style=TEXT,       max_width=40)
    for i, e in enumerate(_history, 1):
        t.add_row(
            str(i), e["mode"], e["ts"],
            (e["input"][:34]  + "…") if len(e["input"])  > 34 else e["input"],
            (e["output"][:38] + "…") if len(e["output"]) > 38 else e["output"],
        )
    console.print(t)
    console.print()


def show_help():
    clear()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {TEXT}", width=12)
    grid.add_column(style=TEXT_DIM)
    for cmd, desc in [
        ("/ttp",     "Toggle voice ↔ text mode"),
        ("/history", "View session history"),
        ("/clear",   "Clear the screen"),
        ("/help",    "Show this message"),
        ("/quit",    "Exit Llama"),
        ("f",        "Refine further (after a result)"),
        ("c",        "Copy result again"),
        ("n",        "New idea"),
        ("q",        "Quit"),
    ]:
        grid.add_row(cmd, desc)
    console.print(Panel(grid, title=f"[{TEXT}]Help[/{TEXT}]", border_style=BORDER, padding=(0, 2)))
    console.print()


# ── Whisper ───────────────────────────────────────────────────────────────────
def load_whisper_model():
    clear()
    console.print(build_header())
    console.print(Columns([
        build_left(),
        Panel(
            Align.center(
                Text("Loading voice model…", style=TEXT_DIM),
                vertical="middle",
            ),
            border_style=BORDER,
        ),
    ], expand=True))
    model = whisper.load_model(WHISPER_MODEL)
    return model


def listen_for_wake_word(whisper_model):
    CHUNK     = SAMPLE_RATE * 4
    triggered = threading.Event()

    clear()
    console.print(build_header())
    console.print(f"\n  [{TEXT_DIM}]Listening for \"hey llama\"… or type it and press Enter[/{TEXT_DIM}]")
    console.print(f"  [{TEXT_FAINT}]──────────────────────────────────────[/{TEXT_FAINT}]\n")

    def wait_enter():
        sys.stdout.write("  › ")
        sys.stdout.flush()
        typed = input().strip().lower()
        if "hey llama" in typed or "hey, llama" in typed:
            console.print(f"\n  [{TEXT}]Wake word detected.[/{TEXT}]")
        triggered.set()

    threading.Thread(target=wait_enter, daemon=True).start()

    while not triggered.is_set():
        audio = sd.rec(CHUNK, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        if triggered.is_set():
            break
        af = audio.astype(np.float32).flatten() / 32768.0
        try:
            text = whisper_model.transcribe(af, language="en", fp16=False)["text"].strip().lower()
            if "llama" in text:
                console.print(f"\n  [{TEXT}]Wake word detected.[/{TEXT}]")
                return
        except Exception:
            pass


def record_audio():
    flush_stdin()
    console.print(f"  [{TEXT_DIM}]Recording… press Enter to stop.[/{TEXT_DIM}]")
    frames = []

    def cb(indata, *_):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=cb):
        input()

    return np.concatenate(frames) if frames else None


def transcribe(audio, model):
    af = audio.astype(np.float32).flatten() / 32768.0
    return model.transcribe(af, language="en")["text"].strip()


# ── Input collection ──────────────────────────────────────────────────────────
def get_idea(whisper_model, active_mode_id=None, last_output=None):
    global _text_mode

    render_screen(None, active_mode_id, last_output)

    if _text_mode:
        console.print(f"\n  [{TEXT_DIM}]Your idea  (/ttp for voice · /help)[/{TEXT_DIM}]")
        flush_stdin()
        raw = input("  › ").strip()
    else:
        console.print(f"\n  [{TEXT_DIM}]Press Enter to record  ·  /ttp for text  ·  /help[/{TEXT_DIM}]")
        flush_stdin()
        resp = input("  › ").strip()

        if resp.lower() == "/ttp":
            _text_mode = True
            return None, whisper_model, True

        if handle_command(resp):
            return None, whisper_model, False

        audio = record_audio()
        if audio is None or len(audio) == 0:
            console.print(f"  [{TEXT_DIM}]No audio detected.[/{TEXT_DIM}]\n")
            return None, whisper_model, False

        with Live(Spinner("dots", text=Text("  Transcribing…", style=TEXT_DIM)),
                  console=console, transient=True):
            raw = transcribe(audio, whisper_model)

        if not raw:
            console.print(f"  [{TEXT_DIM}]No speech detected. Try again.[/{TEXT_DIM}]\n")
            return None, whisper_model, False

    if raw and raw.lower() == "/ttp":
        _text_mode = not _text_mode
        return None, whisper_model, True

    if raw and handle_command(raw):
        return None, whisper_model, False

    return (raw or None), whisper_model, False


def pick_mode(raw_text, active_mode_id=None):
    render_screen(raw_text, active_mode_id)
    console.print(f"\n  [{TEXT_DIM}]Pick a mode:[/{TEXT_DIM}]")
    for key, (mode_id, label, desc) in MODES.items():
        console.print(f"  [{TEXT_FAINT}]{key}[/{TEXT_FAINT}]  [{TEXT_DIM}]{label}[/{TEXT_DIM}]")
    console.print(f"  [{TEXT_FAINT}]5[/{TEXT_FAINT}]  [{TEXT_DIM}]Re-enter idea[/{TEXT_DIM}]")
    console.print()
    flush_stdin()
    return input("  › ").strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _text_mode

    no_wake    = "--no-wake" in sys.argv
    _text_mode = "--text"    in sys.argv

    whisper_model = None
    last_output   = None
    last_mode_id  = None

    if not _text_mode:
        whisper_model = load_whisper_model()
        if not no_wake:
            listen_for_wake_word(whisper_model)

    while True:
        # ── Get idea ────────────────────────────────────────────────────────
        raw, whisper_model, toggled = get_idea(
            whisper_model,
            active_mode_id=last_mode_id,
            last_output=last_output,
        )
        if toggled or not raw:
            continue

        # ── Pick mode ────────────────────────────────────────────────────────
        while True:
            choice = pick_mode(raw, last_mode_id)
            if handle_command(choice):
                continue
            if choice == "5":
                raw = None
                break
            if choice in MODES:
                break
            console.print(f"  [{TEXT_DIM}]Type 1–4.[/{TEXT_DIM}]")

        if raw is None:
            continue

        mode_id      = MODES[choice][0]
        mode_label   = MODES[choice][1]
        last_mode_id = mode_id

        # ── Run + stream ─────────────────────────────────────────────────────
        result      = run_mode(mode_id, raw)
        last_output = result
        copied      = copy_to_clipboard(result)

        _history.append({
            "mode":   mode_label,
            "input":  raw,
            "output": result,
            "ts":     datetime.now().strftime("%H:%M"),
        })

        # ── Post-result loop ─────────────────────────────────────────────────
        current_result = result
        show_copied    = copied

        while True:
            render_screen(raw, mode_id, current_result, copied=show_copied)
            show_copied = False

            console.print(f"\n  [{TEXT_DIM}]f  refine further   c  copy   n  new idea   h  history   q  quit[/{TEXT_DIM}]")
            flush_stdin()
            action = input("  › ").strip().lower()

            if handle_command(action):
                continue

            if action == "f":
                render_screen(raw, mode_id, current_result)
                console.print(f"\n  [{TEXT_DIM}]How should I adjust it?[/{TEXT_DIM}]")
                flush_stdin()
                adjustment = input("  › ").strip()
                if not adjustment:
                    continue
                new_result = run_iterate(current_result, adjustment, mode_id)
                current_result = new_result
                if copy_to_clipboard(new_result):
                    show_copied = True
                _history.append({
                    "mode":   "adjusted",
                    "input":  adjustment,
                    "output": new_result,
                    "ts":     datetime.now().strftime("%H:%M"),
                })

            elif action == "c":
                if copy_to_clipboard(current_result):
                    show_copied = True

            elif action == "h":
                show_history()
                input(f"  Press Enter to continue… ")

            elif action == "n":
                last_output = current_result
                break

            elif action in ("q", "/quit", "/exit"):
                console.print(f"\n[{TEXT_DIM}]Goodbye.[/{TEXT_DIM}]\n")
                sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(f"\n\n[{TEXT_DIM}]Goodbye.[/{TEXT_DIM}]\n")
