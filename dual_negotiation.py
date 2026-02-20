"""
Run a negotiation between two model personas.

All configuration lives in this file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

# =====================
# In-file configuration
# =====================
ENV_FILE = ".env.local"

MODEL_A = "gpt-4.1-mini"
MODEL_B = "gpt-4.1-mini"

NUM_ROUNDS = 12
TEMPERATURE = 0.7
TOPIC = "Forklift resale in Greater Zurich area (price-dominant; currency CHF)."

COMMON_PROMPT_FILE = Path("prompts/common_prompt.txt")
PROMPT_A_FILE = Path("prompts/seller_prompt.txt")
PROMPT_B_FILE = Path("prompts/buyer_prompt.txt")


Message = Dict[str, str]


def load_dotenv_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        # Keep real environment values if already set.
        os.environ.setdefault(key, value)


def load_text_file(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"{label} file is empty: {path}")
    return prompt


def combine_system_prompt(common_prompt: str, role_prompt: str) -> str:
    return f"{common_prompt}\n\n{role_prompt}"


def generate_reply(
    client,
    model: str,
    history: List[Message],
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def run_negotiation(
    client,
    model_a: str,
    model_b: str,
    prompt_a: str,
    prompt_b: str,
    topic: str,
    rounds: int,
    temperature: float,
) -> None:
    history_a: List[Message] = [{"role": "system", "content": prompt_a}]
    history_b: List[Message] = [{"role": "system", "content": prompt_b}]

    latest_from_b = (
        f"Negotiation topic: {topic}\n"
        "Start the conversation with your opening proposal."
    )

    for round_idx in range(1, rounds + 1):
        history_a.append({"role": "user", "content": latest_from_b})
        message_a = generate_reply(client, model_a, history_a, temperature)
        history_a.append({"role": "assistant", "content": message_a})
        print(f"\nRound {round_idx} - Agent A ({model_a}):\n{message_a}")

        to_b = f"Counterparty says:\n{message_a}\nRespond with your best next move."
        history_b.append({"role": "user", "content": to_b})
        message_b = generate_reply(client, model_b, history_b, temperature)
        history_b.append({"role": "assistant", "content": message_b})
        print(f"\nRound {round_idx} - Agent B ({model_b}):\n{message_b}")

        latest_from_b = (
            f"Counterparty says:\n{message_b}\nRespond with your best next move."
        )


def main() -> int:
    if NUM_ROUNDS < 1:
        print("NUM_ROUNDS must be >= 1", file=sys.stderr)
        return 1
    if TEMPERATURE < 0 or TEMPERATURE > 2:
        print("TEMPERATURE must be between 0 and 2", file=sys.stderr)
        return 1

    load_dotenv_file(ENV_FILE)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            f"Missing OPENAI_API_KEY. Set it in {ENV_FILE} or environment.",
            file=sys.stderr,
        )
        return 1

    try:
        common_prompt = load_text_file(COMMON_PROMPT_FILE, "Common prompt")
        prompt_a = combine_system_prompt(
            common_prompt,
            load_text_file(PROMPT_A_FILE, "Agent A prompt"),
        )
        prompt_b = combine_system_prompt(
            common_prompt,
            load_text_file(PROMPT_B_FILE, "Agent B prompt"),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    run_negotiation(
        client=client,
        model_a=MODEL_A,
        model_b=MODEL_B,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        topic=TOPIC,
        rounds=NUM_ROUNDS,
        temperature=TEMPERATURE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
