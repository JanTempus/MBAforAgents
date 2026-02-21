"""
Run a two-agent negotiation using prompt templates and per-player variable files.
"""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

# =====================
# In-file configuration
# =====================
ENV_FILE = ".env.local"

MODEL_A = "gpt-4.1"
MODEL_B = "gpt-4.1"

NUM_ROUNDS = 12
TEMPERATURE = 0.8
TOPIC = "Forklift resale in Greater Zurich area (price-dominant; currency CHF)."

PROMPTS_DIR = Path("prompts/fork_lift_game")
PLAYER_1_PROMPT_FILE = PROMPTS_DIR / "Player_1_prompt.txt"
PLAYER_2_PROMPT_FILE = PROMPTS_DIR / "Player_2_prompt.txt"
PLAYER_1_VARS_FILE = PROMPTS_DIR / "Player_1_prompt.vars"
PLAYER_2_VARS_FILE = PROMPTS_DIR / "Player_2_prompt.vars"

SETTINGS_ASSIGNMENT_TEMPLATE = """Before we begin negotiating, the user wants to know
your agent settings for this run.

The system sampled these values uniformly at random from predefined behavior
levels. You do not choose them. They are fixed and define your behavior:
irrationality_level: {irrationality_level}
win_orientation_level: {win_orientation_level}
trust_level: {trust_level}
deal_need_level: {deal_need_level}

Behavior directives for this run:
- irrationality directive: {irrationality_directive}
- win-orientation directive: {win_orientation_directive}
- trust directive: {trust_directive}
- deal-need directive: {deal_need_directive}

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
    "Zero-Sum Closer (Win-Lose)": (
        "Treat every concession as weakness and focus on maximizing your share. "
        "Use pressure, anchoring, and brinkmanship to win the deal."
    ),
    "Hard Trader (Mostly Win-Lose)": (
        "Prioritize strong outcomes but avoid pure scorched-earth tactics. "
        "Collaborate tactically when useful, otherwise stay positional."
    ),
    "Pragmatic Splitter (Balanced / Deal-First)": (
        "Optimize for agreement and efficiency over dominance. "
        "Concede strategically to keep momentum and reduce risk."
    ),
    "Value Builder (Mostly Win-Win)": (
        "Look for trades that expand value while protecting core interests. "
        "Be transparent about constraints and flexible on structure."
    ),
    "Partnership Architect (Win-Win)": (
        "Treat negotiation as long-term relationship design. "
        "Co-create durable options and prioritize shared success metrics."
    ),
}

TRUST_LEVELS = [
    "you trust nothing; everything they say is a lie",
    "you trust almost nothing and assume deception unless proven otherwise",
    "you are skeptical and verify most claims",
    "you are neutral and require moderate evidence",
    "you are somewhat trusting and accept many claims in good faith",
    "you are very trusting and rarely challenge claims",
    "anything they tell you is a fact",
]

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

DEAL_NEED_DIRECTIVES = {
    "Casual Browser (Nice-to-have)": (
        "Treat the deal as optional and stay relaxed about timing and terms. "
        "Walk away quickly if it is not clean and clearly advantageous."
    ),
    "Selective Improver (Useful upgrade)": (
        "Want the deal if it meaningfully helps, but avoid prolonged friction. "
        "Exit if negotiation costs rise too much."
    ),
    "Committed Optimizer (Strong preference)": (
        "Aim to make it work through several revisions if needed. "
        "Keep a clear walk-away threshold and do not force a bad deal."
    ),
    "Deadline Driver (High urgency)": (
        "Need closure on timeline and actively drive the process forward. "
        "Offer targeted concessions to remove blockers."
    ),
    "Existential Closer (Must-close / Career-risk)": (
        "Treat failure as catastrophic and prioritize signature speed. "
        "Accept imperfection, but do not violate explicit hard constraints."
    ),
}

WIN_ORIENTATION_LEVELS = list(WIN_ORIENTATION_DIRECTIVES.keys())
DEAL_NEED_LEVELS = list(DEAL_NEED_DIRECTIVES.keys())

SEED_MIN = 0
SEED_MAX = 2**63 - 1


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
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def load_text_file(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} file is empty: {path}")
    return text


def load_vars_file(path: Path, label: str) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line in {path}: {raw_line}")
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def render_template(text: str, variables: Dict[str, str], label: str) -> str:
    pattern = re.compile(r"\{\{([A-Z0-9_]+)\}\}")

    missing = {m.group(1) for m in pattern.finditer(text) if m.group(1) not in variables}
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing template vars in {label}: {missing_list}")

    rendered = text
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def generate_reply(client: OpenAI, model: str, history: List[Message], temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def sample_agent_settings(rng: random.Random) -> AgentSettings:
    return {
        "irrationality_level": rng.choice(IRRATIONALITY_LEVELS),
        "win_orientation_level": rng.choice(WIN_ORIENTATION_LEVELS),
        "trust_level": rng.choice(TRUST_LEVELS),
        "deal_need_level": rng.choice(DEAL_NEED_LEVELS),
    }


def sample_distinct_agent_seeds() -> tuple[int, int]:
    seed_rng = random.SystemRandom()
    seed_a = seed_rng.randint(SEED_MIN, SEED_MAX)
    seed_b = seed_rng.randint(SEED_MIN, SEED_MAX)
    while seed_b == seed_a:
        seed_b = seed_rng.randint(SEED_MIN, SEED_MAX)
    return seed_a, seed_b


def format_settings_lines(settings: AgentSettings) -> str:
    return (
        f"irrationality_level: {settings['irrationality_level']}\n"
        f"win_orientation_level: {settings['win_orientation_level']}\n"
        f"trust_level: {settings['trust_level']}\n"
        f"deal_need_level: {settings['deal_need_level']}\n"
        f"irrationality_directive: {IRRATIONALITY_DIRECTIVES[settings['irrationality_level']]}\n"
        f"win_orientation_directive: {WIN_ORIENTATION_DIRECTIVES[settings['win_orientation_level']]}\n"
        f"trust_directive: {TRUST_DIRECTIVES[settings['trust_level']]}\n"
        f"deal_need_directive: {DEAL_NEED_DIRECTIVES[settings['deal_need_level']]}"
    )


def build_settings_assignment_message(settings: AgentSettings) -> str:
    return SETTINGS_ASSIGNMENT_TEMPLATE.format(
        irrationality_level=settings["irrationality_level"],
        win_orientation_level=settings["win_orientation_level"],
        trust_level=settings["trust_level"],
        deal_need_level=settings["deal_need_level"],
        irrationality_directive=IRRATIONALITY_DIRECTIVES[settings["irrationality_level"]],
        win_orientation_directive=WIN_ORIENTATION_DIRECTIVES[settings["win_orientation_level"]],
        trust_directive=TRUST_DIRECTIVES[settings["trust_level"]],
        deal_need_directive=DEAL_NEED_DIRECTIVES[settings["deal_need_level"]],
    )


def assign_and_print_agent_settings(
    history: List[Message],
    agent_label: str,
    model: str,
    settings: AgentSettings,
    seed: int,
) -> None:
    history.append({"role": "user", "content": build_settings_assignment_message(settings)})
    print(
        f"\nAssigned settings - {agent_label} ({model}) [seed={seed}]:\n"
        f"{format_settings_lines(settings)}"
    )


def run_negotiation(
    client: OpenAI,
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
    settings_a = sample_agent_settings(random.Random(seed_a))
    settings_b = sample_agent_settings(random.Random(seed_b))

    assign_and_print_agent_settings(history_a, "Agent A", model_a, settings_a, seed_a)
    assign_and_print_agent_settings(history_b, "Agent B", model_b, settings_b, seed_b)

    latest_from_b = f"Negotiation topic: {topic}\nStart the conversation with your opening proposal."

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

        latest_from_b = f"Counterparty says:\n{message_b}\nRespond with your best next move."


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
        print(f"Missing OPENAI_API_KEY. Set it in {ENV_FILE} or environment.", file=sys.stderr)
        return 1

    try:
        prompt_1_template = load_text_file(PLAYER_1_PROMPT_FILE, "Player 1 prompt")
        prompt_2_template = load_text_file(PLAYER_2_PROMPT_FILE, "Player 2 prompt")
        vars_1 = load_vars_file(PLAYER_1_VARS_FILE, "Player 1 vars")
        vars_2 = load_vars_file(PLAYER_2_VARS_FILE, "Player 2 vars")
        prompt_a = render_template(prompt_1_template, vars_1, "Player 1 prompt")
        prompt_b = render_template(prompt_2_template, vars_2, "Player 2 prompt")
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_negotiation(
        client=OpenAI(api_key=api_key),
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
