"""
Run a negotiation between two model personas.

All configuration lives in this file.
"""

from __future__ import annotations

import os
import random
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

NUM_ROUNDS = 6
TEMPERATURE = 0.8
TOPIC = "Forklift resale in Greater Zurich area (price-dominant; currency CHF)."

# Select prompt set folder inside `prompts/`.
# Example: "fork_lift_game_basic" or "fork_lift_game_rationality"
PROMPT_SET = "fork_lift_game_rationality"

PROMPTS_ROOT = Path("prompts")
PROMPT_SET_DIR = PROMPTS_ROOT / PROMPT_SET
COMMON_PROMPT_FILE = PROMPT_SET_DIR / "common_prompt.txt"
PROMPT_A_FILE = PROMPT_SET_DIR / "seller_prompt.txt"
PROMPT_B_FILE = PROMPT_SET_DIR / "buyer_prompt.txt"

SETTINGS_ASSIGNMENT_TEMPLATE = """Before we begin negotiating, the user wants to know
your agent settings for this run.

The system sampled these values uniformly at random from predefined behavior
levels. You do not choose them. They are fixed and define your behavior:
irrationality_level: {irrationality_level}
win_orientation_level: {win_orientation_level}
trust_level: {trust_level}

Behavior directives for this run:
- irrationality directive: {irrationality_directive}
- win-orientation directive: {win_orientation_directive}
- trust directive: {trust_directive}

Execution requirements:
- Apply these directives consistently in every message.
- Keep respecting your role constraints and hard limits from the scenario.
- Let trust influence tone, disclosure, and concession behavior."""


Message = Dict[str, str]
AgentSettings = Dict[str, str]


IRRATIONALITY_LEVELS = [
    "insanely irrational",
    "very irrational",
    "somewhat irrational",
    "neither irrational nor rational",
    "somewhat rational",
    "very rational",
    "insanely rational",
]

WIN_ORIENTATION_LEVELS = [
    "you do not care if you win or lose",
    "you slightly prefer winning, but losing is acceptable",
    "you somewhat want to win, but can still compromise",
    "you are balanced between winning and reaching a feasible deal",
    "you strongly want to win and prioritize your own outcome",
    "you very strongly need to win and resist compromise",
    "you have to win",
]

TRUST_LEVELS = [
    "you trust nothing; everything they say is a lie",
    "you trust almost nothing and assume deception unless proven otherwise",
    "you are skeptical and verify most claims",
    "you are neutral and require moderate evidence",
    "you are somewhat trusting and accept many claims in good faith",
    "you are very trusting and rarely challenge claims",
    "anything they tell you is a fact",
]

SEED_MIN = 0
SEED_MAX = 2**63 - 1

IRRATIONALITY_DIRECTIVES = {
    "insanely irrational": (
        "make impulsive, inconsistent moves; overreact to small provocations; "
        "occasionally reject clearly good offers or accept poor ones"
    ),
    "very irrational": (
        "frequently misread the situation; make emotion-driven concessions or "
        "demands; show weak internal consistency"
    ),
    "somewhat irrational": (
        "mostly follow logic but sometimes make ego-driven detours, minor "
        "miscalculations, and occasional overreactions"
    ),
    "neither irrational nor rational": (
        "mix logic and emotion evenly; sometimes quantify carefully, sometimes "
        "go with instinct without full justification"
    ),
    "somewhat rational": (
        "usually reason from constraints and incentives, but still allow small "
        "emotional bias in close decisions"
    ),
    "very rational": (
        "be consistent and calculation-driven; prioritize BATNA/ZOPA logic and "
        "avoid emotional swings"
    ),
    "insanely rational": (
        "be maximally systematic and internally consistent; use strict decision "
        "logic, explicit tradeoffs, and zero ego reactions"
    ),
}

WIN_ORIENTATION_DIRECTIVES = {
    "you do not care if you win or lose": (
        "focus on feasibility and completion over personal advantage; accept "
        "balanced outcomes with minimal resistance"
    ),
    "you slightly prefer winning, but losing is acceptable": (
        "seek mild advantage but concede quickly when needed to preserve a "
        "workable agreement"
    ),
    "you somewhat want to win, but can still compromise": (
        "pursue gains while remaining flexible; trade concessions when they "
        "improve chance of closure"
    ),
    "you are balanced between winning and reaching a feasible deal": (
        "equally weight self-interest and deal completion; push where justified "
        "and reciprocate fairly"
    ),
    "you strongly want to win and prioritize your own outcome": (
        "aggressively protect your side; concede only with clear return value "
        "or strong strategic reason"
    ),
    "you very strongly need to win and resist compromise": (
        "treat concessions as costly; escalate demands and only move when "
        "facing concrete risk of breakdown"
    ),
    "you have to win": (
        "prioritize dominance of outcome above relationship and joint value; "
        "concede minimally and only under hard constraints"
    ),
}

TRUST_DIRECTIVES = {
    "you trust nothing; everything they say is a lie": (
        "assume deception by default; challenge nearly every claim and require "
        "strong verification before adjusting position"
    ),
    "you trust almost nothing and assume deception unless proven otherwise": (
        "default to skepticism; verify most statements and delay concessions "
        "until credibility is established"
    ),
    "you are skeptical and verify most claims": (
        "request supporting rationale frequently; accept some claims cautiously "
        "after limited cross-checking"
    ),
    "you are neutral and require moderate evidence": (
        "neither distrust nor over-trust; update belief with moderate evidence "
        "and consistent behavior"
    ),
    "you are somewhat trusting and accept many claims in good faith": (
        "assume mostly good faith; verify only higher-stakes claims and keep "
        "a collaborative tone"
    ),
    "you are very trusting and rarely challenge claims": (
        "treat most statements as credible; prioritize momentum and alignment "
        "over extensive verification"
    ),
    "anything they tell you is a fact": (
        "accept counterpart statements as true unless impossible; adapt quickly "
        "to their framing with minimal challenge"
    ),
}


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


def sample_agent_settings(rng: random.Random) -> AgentSettings:
    return {
        "irrationality_level": rng.choice(IRRATIONALITY_LEVELS),
        "win_orientation_level": rng.choice(WIN_ORIENTATION_LEVELS),
        "trust_level": rng.choice(TRUST_LEVELS),
    }


def format_settings_lines(settings: AgentSettings) -> str:
    return (
        f"irrationality_level: {settings['irrationality_level']}\n"
        f"win_orientation_level: {settings['win_orientation_level']}\n"
        f"trust_level: {settings['trust_level']}\n"
        f"irrationality_directive: "
        f"{IRRATIONALITY_DIRECTIVES[settings['irrationality_level']]}\n"
        f"win_orientation_directive: "
        f"{WIN_ORIENTATION_DIRECTIVES[settings['win_orientation_level']]}\n"
        f"trust_directive: {TRUST_DIRECTIVES[settings['trust_level']]}"
    )


def sample_distinct_agent_seeds() -> tuple[int, int]:
    seed_rng = random.SystemRandom()
    seed_a = seed_rng.randint(SEED_MIN, SEED_MAX)
    seed_b = seed_rng.randint(SEED_MIN, SEED_MAX)
    while seed_b == seed_a:
        seed_b = seed_rng.randint(SEED_MIN, SEED_MAX)
    return seed_a, seed_b


def build_settings_assignment_message(settings: AgentSettings) -> str:
    return SETTINGS_ASSIGNMENT_TEMPLATE.format(
        irrationality_level=settings["irrationality_level"],
        win_orientation_level=settings["win_orientation_level"],
        trust_level=settings["trust_level"],
        irrationality_directive=IRRATIONALITY_DIRECTIVES[
            settings["irrationality_level"]
        ],
        win_orientation_directive=WIN_ORIENTATION_DIRECTIVES[
            settings["win_orientation_level"]
        ],
        trust_directive=TRUST_DIRECTIVES[settings["trust_level"]],
    )


def assign_and_print_agent_settings(
    history: List[Message],
    agent_label: str,
    model: str,
    settings: AgentSettings,
    seed: int,
) -> None:
    history.append(
        {
            "role": "user",
            "content": build_settings_assignment_message(settings),
        }
    )
    print(
        f"\nAssigned settings - {agent_label} ({model}) [seed={seed}]:\n"
        f"{format_settings_lines(settings)}"
    )


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

    seed_a, seed_b = sample_distinct_agent_seeds()
    rng_a = random.Random(seed_a)
    rng_b = random.Random(seed_b)

    settings_a = sample_agent_settings(rng_a)
    settings_b = sample_agent_settings(rng_b)

    assign_and_print_agent_settings(
        history=history_a,
        agent_label="Agent A",
        model=model_a,
        settings=settings_a,
        seed=seed_a,
    )
    assign_and_print_agent_settings(
        history=history_b,
        agent_label="Agent B",
        model=model_b,
        settings=settings_b,
        seed=seed_b,
    )

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
