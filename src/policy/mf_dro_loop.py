#!/usr/bin/env python3
"""
MF-DRO Research Orchestration Loop
====================================
Connects two Claude instances:
  - Research Claude  : interprets experimental results, designs next experiments
  - Agent Claude     : implements code changes, runs diagnostics, reports numbers

Usage:
    python mf_dro_loop.py

Flow:
    1. You paste experimental results (or type a question)
    2. Research Claude interprets and writes an instruction
    3. You confirm (or edit) before sending to Agent
    4. Agent implements and reports back
    5. Repeat

Environment variables required:
    ANTHROPIC_API_KEY  — your Anthropic API key

Session files written to ./sessions/:
    research_history.json  — full Research Claude conversation
    agent_history.json     — full Agent conversation
    transcript.txt         — human-readable log of everything
"""

import os
import sys
import json
import datetime
import textwrap
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("anthropic SDK not found. Run: pip install anthropic")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"

SESSIONS_DIR = Path("./sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TRANSCRIPT_FILE = SESSIONS_DIR / f"transcript_{timestamp}.txt"
RESEARCH_HISTORY_FILE = SESSIONS_DIR / "research_history.json"
AGENT_HISTORY_FILE = SESSIONS_DIR / "agent_history.json"

# ─────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────

# Paste the full research context here (the long context prompt
# you currently paste at the start of each new Research Claude window)
RESEARCH_SYSTEM_PROMPT = """
You are a research advisor for an MF-DRO (Multi-Fidelity Direct Regret Optimization)
implementation project. You analyze experimental results, identify root causes,
form hypotheses, and design targeted experiments.

Core principles:
- Treat all analyses as hypotheses until confirmed by experiment
- Be specific about what each experiment tests and what each outcome means
- State clearly what is confirmed vs assumed
- Design the minimal experiment that answers the key question
- Never recommend running a full Stage 2 experiment when a 10-iteration diagnostic suffices

Current project state:
[PASTE YOUR FULL CONTEXT HERE — the long context block you currently
paste at the start of new sessions. This includes:
- All confirmed fixes (Fix 1-12)
- Benchmark results (Ackley works, Hartmann frozen)
- Root cause analysis so far
- The gradient coherence finding (Hartmann 0.645, Ackley 0.028)
- The AWR result (made things worse early, partial recovery at iter 20)
- The convergence-to-wrong-minimum mechanism]
""".strip()

# Paste your agent's system prompt here
AGENT_SYSTEM_PROMPT = """
You are an implementation agent for the MF-DRO codebase.

Your responsibilities:
- Implement code changes to src/policy/mf_dro.py, src/models/ko_gp.py,
  src/model/decisionTransformer.py as instructed
- Run diagnostic experiments and report exact numbers
- Flag implementation bugs before they corrupt experiments
- Never run Stage 2 or long experiments without explicit instruction

When reporting results, always use this structured format:

RESULT_START
benchmark: <name>
seed: <N>
metric_name: <value>
metric_name: <value>
...
RESULT_END

This makes results parseable by the orchestration script.
""".strip()

# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def load_history(path: Path) -> list:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

def save_history(path: Path, history: list):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def log(label: str, text: str):
    """Append to human-readable transcript."""
    sep = "=" * 70
    entry = f"\n{sep}\n[{datetime.datetime.now().strftime('%H:%M:%S')}] {label}\n{sep}\n{text}\n"
    with open(TRANSCRIPT_FILE, "a") as f:
        f.write(entry)

# ─────────────────────────────────────────────
# API calls
# ─────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_claude(system: str, history: list, user_message: str,
                max_tokens: int = 4096, label: str = "") -> str:
    """Single API call. Appends to history in place and returns reply text."""
    history.append({"role": "user", "content": user_message})
    log(f"USER → {label}", user_message)

    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=history
    )
    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})
    log(f"{label} REPLY", reply)
    return reply

# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

def print_box(title: str, text: str, width: int = 80):
    border = "─" * width
    print(f"\n┌{border}┐")
    print(f"│ {title:<{width-1}}│")
    print(f"├{border}┤")
    for line in text.split("\n"):
        for chunk in textwrap.wrap(line, width - 2) or [""]:
            print(f"│ {chunk:<{width-1}}│")
    print(f"└{border}┘\n")

def get_multiline_input(prompt: str) -> str:
    """Read multi-line input until user types END on its own line."""
    print(f"\n{prompt}")
    print("(Type your input. Enter END on its own line when done.)\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    print("\n" + "═" * 70)
    print("  MF-DRO Research Orchestration Loop")
    print("═" * 70)
    print(f"  Transcript: {TRANSCRIPT_FILE}")
    print(f"  Sessions:   {SESSIONS_DIR}")
    print("\n  Commands:")
    print("    [Enter your result/question, then END]")
    print("    'skip'  — send Research Claude's output to Agent without editing")
    print("    'edit'  — edit Research Claude's output before sending to Agent")
    print("    'agent' — type your own message to Agent (bypass Research Claude)")
    print("    'quit'  — save and exit")
    print("═" * 70 + "\n")

    # Load existing histories (resumes previous session if files exist)
    research_history = load_history(RESEARCH_HISTORY_FILE)
    agent_history    = load_history(AGENT_HISTORY_FILE)

    if research_history:
        print(f"  Resumed research history ({len(research_history)//2} turns)")
    if agent_history:
        print(f"  Resumed agent history ({len(agent_history)//2} turns)\n")

    while True:
        # ── Step 1: Get input from user ──────────────────────────────────
        raw = get_multiline_input("📊 Paste experimental results or ask a question:")

        if raw.strip().lower() == "quit":
            print("\nSaving histories and exiting...")
            save_history(RESEARCH_HISTORY_FILE, research_history)
            save_history(AGENT_HISTORY_FILE, agent_history)
            print(f"Transcript saved to {TRANSCRIPT_FILE}")
            break

        if raw.strip().lower() == "agent":
            # Bypass Research Claude, talk directly to Agent
            agent_msg = get_multiline_input("✏️  Message for Agent:")
            agent_reply = call_claude(
                AGENT_SYSTEM_PROMPT, agent_history,
                agent_msg, max_tokens=8192, label="AGENT"
            )
            print_box("🤖 Agent", agent_reply)
            save_history(AGENT_HISTORY_FILE, agent_history)
            continue

        # ── Step 2: Research Claude interprets ───────────────────────────
        print("\n⏳ Research Claude thinking...\n")
        research_reply = call_claude(
            RESEARCH_SYSTEM_PROMPT, research_history,
            raw, max_tokens=4096, label="RESEARCH"
        )
        save_history(RESEARCH_HISTORY_FILE, research_history)
        print_box("🔬 Research Claude", research_reply)

        # ── Step 3: Confirm before sending to Agent ───────────────────────
        print("Send to Agent? [skip=yes / edit / no / quit]: ", end="")
        action = input().strip().lower()

        if action == "quit":
            save_history(RESEARCH_HISTORY_FILE, research_history)
            save_history(AGENT_HISTORY_FILE, agent_history)
            break

        if action == "no":
            print("Not sent to Agent.\n")
            continue

        if action == "edit":
            print("\nCurrent Research Claude output (copy below):")
            print(research_reply)
            instruction = get_multiline_input("✏️  Edited instruction for Agent:")
        else:
            # "skip" or anything else → send as-is
            instruction = research_reply

        # ── Step 4: Send to Agent ─────────────────────────────────────────
        print("\n⏳ Agent working...\n")
        agent_reply = call_claude(
            AGENT_SYSTEM_PROMPT, agent_history,
            instruction, max_tokens=8192, label="AGENT"
        )
        save_history(AGENT_HISTORY_FILE, agent_history)
        print_box("🤖 Agent", agent_reply)

        # Parse any structured RESULT_START...RESULT_END blocks
        if "RESULT_START" in agent_reply:
            print("\n📋 Parsed results:")
            in_block = False
            for line in agent_reply.split("\n"):
                if line.strip() == "RESULT_START":
                    in_block = True
                    continue
                if line.strip() == "RESULT_END":
                    in_block = False
                    print()
                    continue
                if in_block:
                    print(f"  {line}")

if __name__ == "__main__":
    main()
