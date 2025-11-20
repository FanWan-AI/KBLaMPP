"""LLM-powered synthetic world generator with chunked requests.

The script now issues multiple OpenAI-compatible (DashScope) calls: one to
build the entity catalog and several smaller calls to generate fact
batches.  Chunking improves robustness against truncation/timeouts and
allows us to surface granular progress updates plus raw-response dumps
for each attempt.

Example:

```bash
python offline/gen_world_llm.py \
    --output data/5tuple/synth_world_llm.jsonl \
    --world_json data/raw/synth_world_llm.json \
    --raw_dir data/raw/synth_world_llm \
    --num_entities 18 --num_facts 80 --facts_per_call 16
```

Prerequisites:
- ``pip install openai``
- ``.env`` (or KBLaMPP_ENV_FILE) provides ``DASHSCOPE_API_KEY`` / ``MODEL``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kblampp.utils.gpt_session import LocalGPT

DATE_MIN = date(2005, 1, 1)
DATE_MAX = date(2024, 12, 31)


ENTITY_PROMPT_TEMPLATE = Template(
    """
You are designing a compact but coherent fictional knowledge base for KBLaM++.
Return EXACTLY one JSON object describing the entity catalog only:

{
  "entities": [
     {"id": "E0001", "name": "...", "type": "PERSON|ORG|LOC|EVENT|ROLE|DATE", "summary": "one sentence"},
     ... repeat until you have between ${num_entities_min} and ${num_entities_max} entries ...
  ]
}

Strict rules:
1. IDs increment sequentially from E0001 without gaps.
2. Provide a balanced mix (people, orgs, locations, events, temporal markers, roles).
3. Summaries must stay within one concise sentence with concrete details (dates, locations, motivations).
4. Only output valid JSON. No Markdown fences, no commentary.
"""
)


FACT_PROMPT_TEMPLATE = Template(
    """
You are extending the KB described below.  Here is the entity catalog:

${entity_catalog}

Task: produce EXACTLY ${num_facts} new facts that weave multi-hop chains across these actors.
Return ONLY a JSON object matching the schema:

{
  "facts": [
     {
        "head":    {"id": "E####", "name": "...", "type": "..."},
        "relation":{"name": "..."},
        "tail":    {"id": "E####" or "V####", "name": "...", "type": "PERSON|ORG|LOC|DATE|ROLE|EVENT"},
        "context": {
            "source": "llm_synth",
            "page_title": "head entity name",
            "sent_span": "Single English sentence explicitly stating the fact with dates/locations.",
            "disamb": {"country": "US" or "UK" or "SG", "aliases": [optional aliases]},
            "ver": {"url": null, "rev_id": null}
        },
        "time_window": {
            "start": "YYYY-MM-DD",  # required and MUST be between 2005-01-01 and 2024-12-31 (inclusive)
            "end": "YYYY-MM-DD" or null,
            "source": "text_infer"
        }
     },
     ... exactly ${num_facts} entries ...
  ]
}

Rules:
1. Reuse ONLY the provided entity IDs (no new IDs) and keep facts consistent with summaries.
2. Each fact must be unique, time-stamped (2005-01-01 to 2025-12-31) and human-readable.
3. Include collaborations, funding chains, leadership changes, conference participation, acquisitions, etc., to ensure multi-hop reasoning.
4. Do not repeat identical context sentences; vary phrasing and include concrete numbers/dates when possible.
5. Never repeat the same (head, relation, tail, start-date) combination across facts.
6. Output valid JSON with no trailing commas or explanation text.
"""
)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Return the first valid top-level JSON object substring."""
    stack = []
    start = None
    for idx, ch in enumerate(text):
        if ch == "{" and start is None:
            start = idx
            stack.append(ch)
        elif ch == "{" and start is not None:
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                return text[start : idx + 1]
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(0)
    raise ValueError("Model response did not contain a JSON object")


def log(msg: str) -> None:
    print(f"[gen_world_llm] {msg}", flush=True)


def save_raw(raw_dir: Path, label: str, content: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / label
    path.write_text(content, encoding="utf-8")
    return path


def progress_bar(current: int, total: int, width: int = 32) -> None:
    ratio = 0 if total == 0 else current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[gen_world_llm] Progress [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)


def end_progress() -> None:
    print("", flush=True)


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def date_within_range(day: date | None) -> bool:
    if day is None:
        return False
    return DATE_MIN <= day <= DATE_MAX


def ensure_entity_ids(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, entity in enumerate(entities, start=1):
        entity.setdefault("id", f"E{idx:04d}")
    return entities


def build_entity_catalog_prompt(num_entities: int) -> str:
    return ENTITY_PROMPT_TEMPLATE.substitute(
        num_entities_min=max(10, num_entities - 4),
        num_entities_max=num_entities,
    )


def format_entity_catalog(entities: List[Dict[str, Any]]) -> str:
    lines = []
    for ent in entities:
        summary = ent.get("summary", "")
        lines.append(f"- {ent['id']} | {ent.get('name')} | {ent.get('type')} | {summary}")
    return "\n".join(lines)


def build_fact_prompt(entities: List[Dict[str, Any]], num_facts: int) -> str:
    catalog = format_entity_catalog(entities)
    return FACT_PROMPT_TEMPLATE.substitute(entity_catalog=catalog, num_facts=num_facts)


def validate_fact(fact: Dict[str, Any]) -> None:
    required_top = ["head", "relation", "tail", "context", "time_window"]
    for key in required_top:
        if key not in fact:
            raise ValueError(f"Missing key '{key}' in fact: {fact}")
    for slot_name in ("head", "tail"):
        slot = fact[slot_name]
        if "name" not in slot:
            raise ValueError(f"Slot '{slot_name}' missing name: {slot}")
        if "id" not in slot:
            raise ValueError(f"Slot '{slot_name}' missing id: {slot}")
    relation = fact["relation"]
    if "name" not in relation:
        raise ValueError(f"Relation missing name: {relation}")
    if "sent_span" not in fact["context"]:
        raise ValueError(f"Fact missing context.sent_span: {fact}")
    if "start" not in fact["time_window"]:
        raise ValueError(f"Fact missing time_window.start: {fact}")


def build_entity_lookup(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for ent in entities:
        lookup[ent["id"]] = ent
        name = ent.get("name")
        if name:
            lookup[name.lower()] = ent
    return lookup


def ensure_entity_refs(slot: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> None:
    if slot.get("id") in lookup:
        return
    name = slot.get("name")
    if not name:
        raise ValueError(f"Entity slot missing both id and name: {slot}")
    match = lookup.get(name.lower())
    if not match:
        raise ValueError(f"Could not resolve entity name '{name}' in slot {slot}")
    slot["id"] = match["id"]
    slot.setdefault("type", match.get("type"))


def filter_and_normalize_facts(
    raw_facts: List[Dict[str, Any]],
    relation_start_index: int,
    entity_lookup: Dict[str, Dict[str, Any]],
    seen_keys: Set[Tuple[str, str, str, str]],
) -> Tuple[List[Dict[str, Any]], int, Counter]:
    accepted: List[Dict[str, Any]] = []
    relation_index = relation_start_index
    stats: Counter = Counter()
    for fact in raw_facts:
        try:
            ensure_entity_refs(fact["head"], entity_lookup)
            ensure_entity_refs(fact["tail"], entity_lookup)
            validate_fact(fact)
            relation_name = (fact["relation"].get("name") or "").strip()
            if not relation_name:
                raise ValueError("Relation missing descriptive name")
            start_str = fact["time_window"].get("start")
            start_date = parse_iso_date(start_str)
            if not date_within_range(start_date):
                stats["out_of_range"] += 1
                continue
            key = (fact["head"]["id"], relation_name.lower(), fact["tail"]["id"], start_str)
            if key in seen_keys:
                stats["duplicate"] += 1
                continue
            seen_keys.add(key)
            fact.setdefault("relation", {})
            fact["relation"]["id"] = f"R{relation_index:04d}"
            relation_index += 1
            accepted.append(fact)
        except Exception:
            stats["invalid"] += 1
    return accepted, relation_index, stats


def request_json_object(
    client: LocalGPT,
    prompt: str,
    raw_dir: Path,
    label: str,
    max_attempts: int,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        log(f"{label}: attempt {attempt}/{max_attempts}")
        response = client.generate_response(prompt)
        raw_path = save_raw(raw_dir, f"{label.replace(' ', '_')}_attempt{attempt}.txt", response)
        try:
            json_text = extract_json_block(response)
            parsed = json.loads(json_text)
            return parsed
        except Exception as err:  # pragma: no cover - best-effort parsing
            last_error = err
            log(f"{label}: parsing failed ({err}); raw saved to {raw_path}")
    raise RuntimeError(f"{label}: exhausted {max_attempts} attempts") from last_error


def generate_entities(client: LocalGPT, args: argparse.Namespace, raw_dir: Path) -> List[Dict[str, Any]]:
    prompt = build_entity_catalog_prompt(args.num_entities)
    parsed = request_json_object(client, prompt, raw_dir, "entities", args.max_attempts)
    entities = parsed.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("Entity response did not contain a non-empty 'entities' list")
    entities = ensure_entity_ids(entities)
    log(f"entities: collected {len(entities)} entries")
    return entities


def generate_fact_batches(
    client: LocalGPT,
    entities: List[Dict[str, Any]],
    args: argparse.Namespace,
    raw_dir: Path,
) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    entity_lookup = build_entity_lookup(entities)
    relation_index = 1
    total = args.num_facts
    chunk = 0
    seen_keys: Set[Tuple[str, str, str, str]] = set()
    current_batch_size = min(args.facts_per_call, total)
    min_batch_size = min(args.min_facts_per_call, total)
    try:
        while len(facts) < total:
            remaining = total - len(facts)
            batch_size = min(current_batch_size, remaining)
            label = f"facts_batch{chunk + 1:02d}_n{batch_size:02d}"
            prompt = build_fact_prompt(entities, batch_size)
            try:
                parsed = request_json_object(client, prompt, raw_dir, label, args.max_attempts)
            except RuntimeError as err:
                if batch_size <= min_batch_size:
                    raise
                current_batch_size = max(min_batch_size, max(1, batch_size // 2))
                log(
                    f"{label}: failed after {args.max_attempts} attempts, reducing batch size to {current_batch_size} and retrying"
                )
                continue
            chunk += 1
            batch = parsed.get("facts")
            if not isinstance(batch, list) or not batch:
                raise ValueError(f"{label}: response missing 'facts' list")
            if len(batch) != batch_size:
                log(f"{label}: expected {batch_size} facts, received {len(batch)}; truncating/examining")
                batch = batch[:batch_size]
            normalized, relation_index, stats = filter_and_normalize_facts(
                batch,
                relation_index,
                entity_lookup,
                seen_keys,
            )
            rejected = sum(stats.values())
            if rejected:
                details = ", ".join(f"{key}={value}" for key, value in stats.items())
                log(f"{label}: rejected {rejected} facts ({details})")
            if not normalized:
                if current_batch_size > min_batch_size:
                    current_batch_size = max(min_batch_size, max(1, batch_size // 2))
                    log(
                        f"{label}: no usable facts accepted; reducing batch size to {current_batch_size} and retrying"
                    )
                    continue
                log(f"{label}: no usable facts accepted even at minimum batch size; retrying")
                continue
            facts.extend(normalized)
            chunk += 1
            progress_bar(len(facts), total, width=args.progress_width)
    finally:
        end_progress()
    log(f"facts: collected {len(facts)} entries across {chunk} batches")
    return facts


def write_outputs(
    entities: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    world_json: Path,
    output_jsonl: Path,
) -> None:
    world = {"entities": entities, "facts": facts}
    world_json.parent.mkdir(parents=True, exist_ok=True)
    world_json.write_text(json.dumps(world, ensure_ascii=False, indent=2), encoding="utf-8")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for fact in facts:
            fout.write(json.dumps(fact, ensure_ascii=False))
            fout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KB five-tuples via chunked LLM calls")
    parser.add_argument("--output", default="data/5tuple/synth_world_llm.jsonl", help="Target JSONL file for five-tuples")
    parser.add_argument("--world_json", default="data/raw/synth_world_llm.json", help="Optional file to store the full world JSON")
    parser.add_argument("--raw_dir", default="data/raw/synth_world_llm", help="Directory to store raw LLM responses")
    parser.add_argument("--num_entities", type=int, default=18, help="Number of entities to request (upper bound)")
    parser.add_argument("--num_facts", type=int, default=80, help="Total number of five-tuples to synthesize")
    parser.add_argument("--facts_per_call", type=int, default=16, help="Number of facts per LLM request (smaller -> safer)")
    parser.add_argument(
        "--min_facts_per_call",
        type=int,
        default=4,
        help="Smallest batch size when auto-splitting after repeated failures",
    )
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--endpoint", default=None, help="Override endpoint URL")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_attempts", type=int, default=3, help="Retries per call when parsing fails")
    parser.add_argument("--progress_width", type=int, default=32, help="Progress bar width")
    args = parser.parse_args()

    if args.facts_per_call <= 0:
        raise ValueError("--facts_per_call must be positive")
    if args.num_facts <= 0:
        raise ValueError("--num_facts must be positive")
    if args.min_facts_per_call <= 0:
        raise ValueError("--min_facts_per_call must be positive")
    if args.min_facts_per_call > args.facts_per_call:
        raise ValueError("--min_facts_per_call cannot exceed --facts_per_call")
    if args.progress_width <= 0:
        raise ValueError("--progress_width must be positive")

    raw_dir = Path(args.raw_dir)

    log(
        f"Init: entities~{args.num_entities}, total facts={args.num_facts}, batch={args.facts_per_call}, model={args.model or 'auto'}"
    )
    client = LocalGPT(
        model_name=args.model,
        endpoint_url=args.endpoint,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    log(f"Connected to model={client.model_name} at {client.base_url}")

    entities = generate_entities(client, args, raw_dir)
    facts = generate_fact_batches(client, entities, args, raw_dir)

    write_outputs(entities, facts, Path(args.world_json), Path(args.output))
    log(
        f"Done: wrote {len(facts)} facts -> {args.output}, world JSON -> {args.world_json}. Raw attempts under {raw_dir}"
    )


if __name__ == "__main__":
    main()
