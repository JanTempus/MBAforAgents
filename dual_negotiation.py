"""
Run a two-agent negotiation using prompt templates and per-player variable files.
"""

from __future__ import annotations

import os
import random
import re
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# =====================
# In-file configuration
# =====================
ENV_FILE = ".env.local"

MODEL_A = "gpt-4.1-mini"
MODEL_B = "gpt-4.1-mini"

NUM_ROUNDS = 12
TEMPERATURE = 0.8
TOPIC = "Forklift resale in Greater Zurich area (price-dominant; currency CHF)."

PROMPTS_DIR = Path("prompts/fork_lift_game")
PLAYER_1_PROMPT_FILE = PROMPTS_DIR / "Player_1_prompt.txt"
PLAYER_2_PROMPT_FILE = PROMPTS_DIR / "Player_2_prompt.txt"
PLAYER_1_VARS_FILE = PROMPTS_DIR / "Player_1_prompt.vars"
PLAYER_2_VARS_FILE = PROMPTS_DIR / "Player_2_prompt.vars"
ANALYSIS_PROMPTS_DIR = Path("prompts/analysis")
EXTRACT_FINAL_DEAL_PROMPT_FILE = ANALYSIS_PROMPTS_DIR / "extract_final_deal.txt"
ASSESS_AGENT_BEHAVIOR_PROMPT_FILE = ANALYSIS_PROMPTS_DIR / "assess_agent_behavior.txt"

AGENT_SETTINGS_DIR = Path("prompts/agent_settings")
SETTINGS_ASSIGNMENT_TEMPLATE_FILE = AGENT_SETTINGS_DIR / "settings_assignment_template.txt"

DIRECTIVES_DIR = Path("directives")
IRRATIONALITY_DIRECTIVES_FILE = DIRECTIVES_DIR / "irrationality.json"
WIN_ORIENTATION_DIRECTIVES_FILE = DIRECTIVES_DIR / "win_orientation.json"
TRUST_DIRECTIVES_FILE = DIRECTIVES_DIR / "trust.json"
DEAL_NEED_DIRECTIVES_FILE = DIRECTIVES_DIR / "deal_need.json"

STRUCTURED_OUTPUTS_DIR = Path("structured_outputs")
FINAL_DEAL_SCHEMA_FILE = STRUCTURED_OUTPUTS_DIR / "final_deal_extraction.schema.json"
BEHAVIOR_SCHEMA_FILE = STRUCTURED_OUTPUTS_DIR / "behavior_assessment.schema.json"

OUTPUT_DIR = Path("outputs")

ANALYSIS_MODEL = "gpt-4.1"
ANALYSIS_TEMPERATURE = 0.0

Message = Dict[str, str]
AgentSettings = Dict[str, str]
SETTINGS_ASSIGNMENT_TEMPLATE = ""

IRRATIONALITY_DIRECTIVES: Dict[str, str] = {}
WIN_ORIENTATION_DIRECTIVES: Dict[str, str] = {}
TRUST_DIRECTIVES: Dict[str, str] = {}
DEAL_NEED_DIRECTIVES: Dict[str, str] = {}

IRRATIONALITY_LEVELS: List[str] = []
WIN_ORIENTATION_LEVELS: List[str] = []
TRUST_LEVELS: List[str] = []
DEAL_NEED_LEVELS: List[str] = []

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


def load_directives_file(path: Path, label: str) -> Dict[str, str]:
    text = load_text_file(path, label)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{label} must be a non-empty JSON object: {path}")

    directives: Dict[str, str] = {}
    for raw_key, raw_value in data.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{label} has an invalid empty/non-string key in {path}")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{label} has an invalid empty/non-string value for '{raw_key}' in {path}")
        directives[raw_key] = raw_value
    return directives


def load_runtime_assets() -> None:
    global SETTINGS_ASSIGNMENT_TEMPLATE
    global IRRATIONALITY_DIRECTIVES
    global WIN_ORIENTATION_DIRECTIVES
    global TRUST_DIRECTIVES
    global DEAL_NEED_DIRECTIVES
    global IRRATIONALITY_LEVELS
    global WIN_ORIENTATION_LEVELS
    global TRUST_LEVELS
    global DEAL_NEED_LEVELS

    SETTINGS_ASSIGNMENT_TEMPLATE = load_text_file(
        SETTINGS_ASSIGNMENT_TEMPLATE_FILE,
        "Settings assignment template",
    )

    IRRATIONALITY_DIRECTIVES = load_directives_file(
        IRRATIONALITY_DIRECTIVES_FILE,
        "Irrationality directives",
    )
    WIN_ORIENTATION_DIRECTIVES = load_directives_file(
        WIN_ORIENTATION_DIRECTIVES_FILE,
        "Win orientation directives",
    )
    TRUST_DIRECTIVES = load_directives_file(
        TRUST_DIRECTIVES_FILE,
        "Trust directives",
    )
    DEAL_NEED_DIRECTIVES = load_directives_file(
        DEAL_NEED_DIRECTIVES_FILE,
        "Deal need directives",
    )

    IRRATIONALITY_LEVELS = list(IRRATIONALITY_DIRECTIVES.keys())
    WIN_ORIENTATION_LEVELS = list(WIN_ORIENTATION_DIRECTIVES.keys())
    TRUST_LEVELS = list(TRUST_DIRECTIVES.keys())
    DEAL_NEED_LEVELS = list(DEAL_NEED_DIRECTIVES.keys())

    if not IRRATIONALITY_LEVELS:
        raise ValueError("No irrationality levels loaded from directives.")
    if not WIN_ORIENTATION_LEVELS:
        raise ValueError("No win orientation levels loaded from directives.")
    if not TRUST_LEVELS:
        raise ValueError("No trust levels loaded from directives.")
    if not DEAL_NEED_LEVELS:
        raise ValueError("No deal need levels loaded from directives.")


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


def format_options_for_prompt(options: List[str]) -> str:
    return "\n".join(f"- {opt}" for opt in options)


def format_run_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%M,%H,%d,%m,%Y")


def deep_replace_schema_tokens(node: Any, replacements: Dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {k: deep_replace_schema_tokens(v, replacements) for k, v in node.items()}
    if isinstance(node, list):
        if len(node) == 1 and isinstance(node[0], str) and node[0] in replacements:
            replacement = replacements[node[0]]
            if isinstance(replacement, list):
                return replacement
        return [deep_replace_schema_tokens(item, replacements) for item in node]
    if isinstance(node, str) and node in replacements:
        replacement = replacements[node]
        if isinstance(replacement, str):
            return replacement
    return node


def load_structured_schema(
    path: Path,
    label: str,
    replacements: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    text = load_text_file(path, label)
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must contain a JSON object schema.")
    if replacements:
        schema = deep_replace_schema_tokens(schema, replacements)
    return schema


def generate_reply(client: OpenAI, model: str, history: List[Message], temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def generate_structured_reply(
    client: OpenAI,
    model: str,
    history: List[Message],
    temperature: float,
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("Model returned empty JSON content.")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Model JSON response must be an object.")
    return data


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


def assign_agent_settings(
    history: List[Message],
    settings: AgentSettings,
) -> None:
    history.append({"role": "user", "content": build_settings_assignment_message(settings)})


def build_transcript_text(
    rounds_data: List[Tuple[int, str, str, str]],
) -> str:
    lines: List[str] = []
    for round_idx, agent_label, model, message in rounds_data:
        lines.append(f"Round {round_idx} | {agent_label} ({model}):")
        lines.append(message)
        lines.append("")
    return "\n".join(lines).strip()


def build_behavior_assessment_schema(agent_label: str) -> Dict[str, Any]:
    return load_structured_schema(
        path=BEHAVIOR_SCHEMA_FILE,
        label="Behavior assessment schema",
        replacements={
            "__AGENT_LABEL__": agent_label,
            "__IRRATIONALITY_LEVELS__": IRRATIONALITY_LEVELS,
            "__WIN_ORIENTATION_LEVELS__": WIN_ORIENTATION_LEVELS,
            "__TRUST_LEVELS__": TRUST_LEVELS,
            "__DEAL_NEED_LEVELS__": DEAL_NEED_LEVELS,
        },
    )


def extract_final_deal(client: OpenAI, transcript_text: str) -> Dict[str, Any]:
    prompt_template = load_text_file(
        EXTRACT_FINAL_DEAL_PROMPT_FILE,
        "Extract final deal prompt",
    )
    user_prompt = render_template(
        prompt_template,
        {"TRANSCRIPT_TEXT": transcript_text},
        "Extract final deal prompt",
    )
    history: List[Message] = [
        {
            "role": "system",
            "content": (
                "You extract negotiation outcomes from transcripts. "
                "Return only data that matches the provided structured output schema."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return generate_structured_reply(
        client=client,
        model=ANALYSIS_MODEL,
        history=history,
        temperature=ANALYSIS_TEMPERATURE,
        schema_name="final_deal_extraction",
        schema=load_structured_schema(
            FINAL_DEAL_SCHEMA_FILE,
            "Final deal extraction schema",
        ),
    )


def assess_agent_behavior(
    client: OpenAI,
    transcript_text: str,
    agent_label: str,
) -> Dict[str, Any]:
    prompt_template = load_text_file(
        ASSESS_AGENT_BEHAVIOR_PROMPT_FILE,
        "Assess agent behavior prompt",
    )
    user_prompt = render_template(
        prompt_template,
        {
            "AGENT_LABEL": agent_label,
            "IRRATIONALITY_OPTIONS": format_options_for_prompt(IRRATIONALITY_LEVELS),
            "WIN_ORIENTATION_OPTIONS": format_options_for_prompt(WIN_ORIENTATION_LEVELS),
            "TRUST_OPTIONS": format_options_for_prompt(TRUST_LEVELS),
            "DEAL_NEED_OPTIONS": format_options_for_prompt(DEAL_NEED_LEVELS),
            "TRANSCRIPT_TEXT": transcript_text,
        },
        "Assess agent behavior prompt",
    )
    history: List[Message] = [
        {
            "role": "system",
            "content": (
                "You are an impartial evaluator. Judge observable behavior only. "
                "Do not assume hidden configuration values. "
                "Return only data that matches the provided structured output schema."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return generate_structured_reply(
        client=client,
        model=ANALYSIS_MODEL,
        history=history,
        temperature=ANALYSIS_TEMPERATURE,
        schema_name=f"behavior_assessment_{agent_label.lower().replace(' ', '_')}",
        schema=build_behavior_assessment_schema(agent_label),
    )


def write_settings_markdown(
    path: Path,
    run_timestamp: str,
    topic: str,
    rounds: int,
    model_a: str,
    model_b: str,
    seed_a: int,
    seed_b: int,
    settings_a: AgentSettings,
    settings_b: AgentSettings,
) -> None:
    content = (
        "# Negotiation Settings\n\n"
        f"- Timestamp (Minute,hour,day,month,year): {run_timestamp}\n"
        f"- Topic: {topic}\n"
        f"- Rounds: {rounds}\n"
        f"- Agent A model: {model_a}\n"
        f"- Agent B model: {model_b}\n\n"
        "## Agent A\n\n"
        f"- Seed: {seed_a}\n\n"
        "```text\n"
        f"{format_settings_lines(settings_a)}\n"
        "```\n\n"
        "## Agent B\n\n"
        f"- Seed: {seed_b}\n\n"
        "```text\n"
        f"{format_settings_lines(settings_b)}\n"
        "```\n"
    )
    path.write_text(content, encoding="utf-8")


def write_transcript_markdown(
    path: Path,
    run_timestamp: str,
    topic: str,
    rounds_data: List[Tuple[int, str, str, str]],
    final_deal_json: Dict[str, Any],
    agent_a_eval_json: Dict[str, Any],
    agent_b_eval_json: Dict[str, Any],
) -> None:
    lines: List[str] = [
        "# Negotiation Transcript",
        "",
        f"- Timestamp (Minute,hour,day,month,year): {run_timestamp}",
        f"- Topic: {topic}",
        "",
        "## Discussion",
        "",
    ]
    for round_idx, agent_label, model, message in rounds_data:
        lines.append(f"### Round {round_idx} - {agent_label} ({model})")
        lines.append("")
        lines.append(message)
        lines.append("")

    lines.extend(
        [
            "## Final Deal Extraction",
            "",
            "```json",
            json.dumps(final_deal_json, indent=2),
            "```",
            "",
            "## Behavior Assessment: Agent A",
            "",
            "```json",
            json.dumps(agent_a_eval_json, indent=2),
            "```",
            "",
            "## Behavior Assessment: Agent B",
            "",
            "```json",
            json.dumps(agent_b_eval_json, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_negotiation(
    client: OpenAI,
    model_a: str,
    model_b: str,
    prompt_a: str,
    prompt_b: str,
    topic: str,
    rounds: int,
    temperature: float,
) -> Dict[str, object]:
    history_a: List[Message] = [{"role": "system", "content": prompt_a}]
    history_b: List[Message] = [{"role": "system", "content": prompt_b}]
    rounds_data: List[Tuple[int, str, str, str]] = []

    seed_a, seed_b = sample_distinct_agent_seeds()
    settings_a = sample_agent_settings(random.Random(seed_a))
    settings_b = sample_agent_settings(random.Random(seed_b))

    assign_agent_settings(history_a, settings_a)
    assign_agent_settings(history_b, settings_b)

    latest_from_b = f"Negotiation topic: {topic}\nStart the conversation with your opening proposal."

    for round_idx in range(1, rounds + 1):
        history_a.append({"role": "user", "content": latest_from_b})
        message_a = generate_reply(client, model_a, history_a, temperature)
        history_a.append({"role": "assistant", "content": message_a})
        rounds_data.append((round_idx, "Agent A", model_a, message_a))

        to_b = f"Counterparty says:\n{message_a}\nRespond with your best next move."
        history_b.append({"role": "user", "content": to_b})
        message_b = generate_reply(client, model_b, history_b, temperature)
        history_b.append({"role": "assistant", "content": message_b})
        rounds_data.append((round_idx, "Agent B", model_b, message_b))

        latest_from_b = f"Counterparty says:\n{message_b}\nRespond with your best next move."

    transcript_text = build_transcript_text(rounds_data)
    final_deal_json = extract_final_deal(client, transcript_text)
    agent_a_eval_json = assess_agent_behavior(client, transcript_text, "Agent A")
    agent_b_eval_json = assess_agent_behavior(client, transcript_text, "Agent B")

    return {
        "seed_a": seed_a,
        "seed_b": seed_b,
        "settings_a": settings_a,
        "settings_b": settings_b,
        "rounds_data": rounds_data,
        "final_deal_json": final_deal_json,
        "agent_a_eval_json": agent_a_eval_json,
        "agent_b_eval_json": agent_b_eval_json,
    }


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
        load_runtime_assets()
        prompt_1_template = load_text_file(PLAYER_1_PROMPT_FILE, "Player 1 prompt")
        prompt_2_template = load_text_file(PLAYER_2_PROMPT_FILE, "Player 2 prompt")
        vars_1 = load_vars_file(PLAYER_1_VARS_FILE, "Player 1 vars")
        vars_2 = load_vars_file(PLAYER_2_VARS_FILE, "Player 2 vars")
        prompt_a = render_template(prompt_1_template, vars_1, "Player 1 prompt")
        prompt_b = render_template(prompt_2_template, vars_2, "Player 2 prompt")
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_timestamp = format_run_timestamp(datetime.now())

    result = run_negotiation(
        client=OpenAI(api_key=api_key),
        model_a=MODEL_A,
        model_b=MODEL_B,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        topic=TOPIC,
        rounds=NUM_ROUNDS,
        temperature=TEMPERATURE,
    )

    run_output_dir = OUTPUT_DIR / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    settings_md_file = run_output_dir / "settings.md"
    transcript_md_file = run_output_dir / "transcript.md"
    write_settings_markdown(
        path=settings_md_file,
        run_timestamp=run_timestamp,
        topic=TOPIC,
        rounds=NUM_ROUNDS,
        model_a=MODEL_A,
        model_b=MODEL_B,
        seed_a=int(result["seed_a"]),
        seed_b=int(result["seed_b"]),
        settings_a=result["settings_a"],  # type: ignore[arg-type]
        settings_b=result["settings_b"],  # type: ignore[arg-type]
    )
    write_transcript_markdown(
        path=transcript_md_file,
        run_timestamp=run_timestamp,
        topic=TOPIC,
        rounds_data=result["rounds_data"],  # type: ignore[arg-type]
        final_deal_json=result["final_deal_json"],  # type: ignore[arg-type]
        agent_a_eval_json=result["agent_a_eval_json"],  # type: ignore[arg-type]
        agent_b_eval_json=result["agent_b_eval_json"],  # type: ignore[arg-type]
    )

    print(f"Timestamp (Minute,hour,day,month,year): {run_timestamp}")
    print(f"Run folder: {run_output_dir}")
    print(f"Settings written to: {settings_md_file}")
    print(f"Transcript written to: {transcript_md_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
