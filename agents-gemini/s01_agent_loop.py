#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop (Gemini API)

The entire secret of an AI coding agent in one pattern:

    while model returns function calls:
        response = Gemini(contents, tools)
        execute tools
        append function results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          | function_response |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model stops calling tools. Production agents layer
policy, hooks, and lifecycle controls on top.

Env:
  GEMINI_API_KEY or GOOGLE_API_KEY — API key
  MODEL_ID — model name (e.g. gemini-2.0-flash)
  Optional: GEMINI_BASE_URL — override API base URL (HttpOptions.base_url)
"""

import os
import subprocess

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")

_http_opts = None
_base = os.getenv("GEMINI_BASE_URL")
if _base:
    _http_opts = types.HttpOptions(base_url=_base)

client = genai.Client(api_key=_api_key, http_options=_http_opts)
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

BASH_DECL = types.FunctionDeclaration(
    name="bash",
    description="Run a shell command.",
    parameters_json_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)
BASH_TOOL = types.Tool(function_declarations=[BASH_DECL])


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(contents: list[types.Content]) -> None:
    while True:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                tools=[BASH_TOOL],
                max_output_tokens=8000,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        if not response.candidates or response.candidates[0].content is None:
            return
        model_content = response.candidates[0].content
        contents.append(model_content)
        fcalls = response.function_calls
        if not fcalls:
            return
        fr_parts: list[types.Part] = []
        for fc in fcalls:
            name = fc.name or "bash"
            args = fc.args or {}
            if name == "bash":
                cmd = args.get("command", "")
                print(f"\033[33m$ {cmd}\033[0m")
                output = run_bash(str(cmd))
                print(output[:200])
                payload: dict = {"result": output}
            else:
                payload = {"error": f"unknown tool: {name}"}
            fr = types.FunctionResponse(
                id=fc.id,
                name=name,
                response=payload,
            )
            fr_parts.append(types.Part(function_response=fr))
        contents.append(types.Content(role="tool", parts=fr_parts))


def _print_model_text(last: types.Content) -> None:
    if last.role != "model" or not last.parts:
        return
    for part in last.parts:
        if part.text:
            print(part.text)


if __name__ == "__main__":
    history: list[types.Content] = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)],
            )
        )
        agent_loop(history)
        _print_model_text(history[-1])
        print()
