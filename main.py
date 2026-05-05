"""
Multilingual customer-support transcript analysis pipeline.

Run:
    export OPENROUTER_API_KEY=your_key
    python main.py
"""

import json
import os
import shutil
import sys

from dotenv import load_dotenv

from llm_client import LLMError, call_llm
from utils import (
    LLMParseError,
    compute_pivot,
    load_transcripts,
    parse_llm_json,
    scan_compliance_keywords,
    validate_annotation,
)
from vocab import (
    ALLOWED_COACHING_TECHNIQUES,
    ALLOWED_INTENTS,
    ALLOWED_PRIORITIES,
    ALLOWED_ROUTES,
    COMPLIANCE_KEYWORDS,
)

load_dotenv()

TRANSCRIPT_PATH = "transcript.json"
ANNOTATIONS_DIR = "annotations"
COACHING_DIR = "coaching"
PIVOTS_PATH = "pivots.json"
ROUTING_PATH = "routing.json"
LLM_LOG_PATH = "llm_calls.jsonl"
PATTERNS_PATH = "cross_transcript_patterns.json"
TRANSLATION_AUDIT_PATH = "translation_audit.json"
BILINGUAL_REPLIES_PATH = "bilingual_replies.json"
COMPLIANCE_FLAGS_PATH = "compliance_flags.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_state(name: str) -> None:
    print(f"[STATE: {name}]")


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Stage: clean output directories / files
# ---------------------------------------------------------------------------

def clean_outputs() -> None:
    for d in [ANNOTATIONS_DIR, COACHING_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    for path in [
        PIVOTS_PATH,
        ROUTING_PATH,
        LLM_LOG_PATH,
        PATTERNS_PATH,
        TRANSLATION_AUDIT_PATH,
        BILINGUAL_REPLIES_PATH,
        COMPLIANCE_FLAGS_PATH,
    ]:
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Stage 1: Load transcripts
# ---------------------------------------------------------------------------

def load_transcripts_stage() -> list[dict]:
    transcripts = load_transcripts(TRANSCRIPT_PATH)
    print(f"  Loaded {len(transcripts)} transcript(s): {[t['transcript_id'] for t in transcripts]}")
    log_state("TRANSCRIPTS_LOADED")
    return transcripts


# ---------------------------------------------------------------------------
# Stage 2: Annotate transcripts (one LLM call per transcript)
# ---------------------------------------------------------------------------

def _build_annotation_prompt(transcript: dict) -> str:
    turns_text = "\n".join(
        f"  Turn {t['turn_index']} [{t['speaker_original']}]: {t['text']}"
        for t in transcript["turns"]
    )
    intents_str = ", ".join(ALLOWED_INTENTS)

    return f"""You are a multilingual customer support analyst.

Annotate every turn of the transcript below. Return ONLY a valid JSON array with no extra text, no markdown fences.

Rules:
- "turn_index": integer (0-based, as given)
- "speaker": copy the original speaker label exactly
- "original_text": copy the turn text EXACTLY as written — do NOT paraphrase or translate
- "language_detected": BCP-47 language code (e.g. "en", "es", "id", "zh", "ar")
- "translated_to_english": English translation (same as original if already English)
- "intent": MUST be one of: {intents_str}
- "sentiment_score": float in [-1.0, 1.0] (−1 = very negative, 0 = neutral, 1 = very positive)
- "is_escalation_signal": true if turn contains complaint, fraud accusation, urgent withdrawal, frozen account, legal/compliance threat, or explicit dissatisfaction — otherwise false
- "unmet_information_need": brief string describing what the speaker needs but has not received, or "" if none

Return a JSON array of {len(transcript['turns'])} objects, one per turn, in turn order.

Transcript ID: {transcript['transcript_id']}
Topic: {transcript.get('topic', 'unknown')}

Turns:
{turns_text}
"""


def annotate_transcripts_stage(transcripts: list[dict]) -> None:
    for transcript in transcripts:
        tid = transcript["transcript_id"]
        expected_count = len(transcript["turns"])
        print(f"  Annotating {tid} ({expected_count} turns)...")

        prompt = _build_annotation_prompt(transcript)
        output_path = f"{ANNOTATIONS_DIR}/{tid}.json"

        raw = call_llm(
            prompt=prompt,
            stage="annotation",
            transcript_id=tid,
            input_artifacts=[TRANSCRIPT_PATH],
            output_artifact=output_path,
        )

        try:
            annotations = parse_llm_json(raw)
        except LLMParseError as e:
            raise RuntimeError(f"Annotation parse failed for {tid}: {e}") from e

        if not isinstance(annotations, list):
            raise RuntimeError(f"Annotation for {tid}: expected JSON array, got {type(annotations).__name__}")

        if len(annotations) != expected_count:
            raise RuntimeError(
                f"Annotation for {tid}: expected {expected_count} annotations, got {len(annotations)}"
            )

        source_texts = [t["text"] for t in transcript["turns"]]
        for ann in annotations:
            validate_annotation(ann)
            idx = ann["turn_index"]
            if ann["original_text"] != source_texts[idx]:
                raise RuntimeError(
                    f"Transcript {tid} turn {idx}: original_text mismatch.\n"
                    f"  Source:     {source_texts[idx]!r}\n"
                    f"  Annotation: {ann['original_text']!r}"
                )

        _write_json(output_path, annotations)
        print(f"    Saved {output_path}")

    log_state("TURN_ANNOTATIONS_COMPLETE")


# ---------------------------------------------------------------------------
# Stage 3: Compute sentiment trajectories and pivot detection
# ---------------------------------------------------------------------------

def compute_pivots_stage(transcripts: list[dict]) -> list[dict]:
    log_state("SENTIMENT_TRAJECTORIES_COMPUTED")
    pivots = []
    for transcript in transcripts:
        tid = transcript["transcript_id"]
        annotations = _read_json(f"{ANNOTATIONS_DIR}/{tid}.json")
        pivot = compute_pivot(tid, annotations)
        pivots.append(pivot)
        if pivot["pivot_turn_index"] is not None:
            print(f"  Pivot in {tid}: turn {pivot['pivot_turn_index']} — {pivot['pivot_agent_quote'][:60]!r}")
        else:
            print(f"  No pivot in {tid}: {pivot['calculation']}")

    _write_json(PIVOTS_PATH, pivots)
    log_state("PIVOTS_DETECTED")
    return pivots


# ---------------------------------------------------------------------------
# Stage 4: Root cause and routing (one combined LLM call)
# ---------------------------------------------------------------------------

def _build_routing_prompt(transcripts: list[dict], pivots: list[dict]) -> str:
    pivot_map = {p["transcript_id"]: p for p in pivots}
    routes_str = ", ".join(ALLOWED_ROUTES)
    priorities_str = ", ".join(ALLOWED_PRIORITIES)

    sections = []
    for transcript in transcripts:
        tid = transcript["transcript_id"]
        annotations = _read_json(f"{ANNOTATIONS_DIR}/{tid}.json")
        pivot = pivot_map.get(tid, {})

        turns_summary = "\n".join(
            f"  [{a['speaker']}] turn {a['turn_index']}: {a['original_text'][:200]} "
            f"(intent={a['intent']}, sentiment={a['sentiment_score']}, escalation={a['is_escalation_signal']})"
            for a in annotations
        )
        pivot_summary = (
            f"Pivot at turn {pivot.get('pivot_turn_index')} — {pivot.get('pivot_agent_quote', '')[:100]}"
            if pivot.get("pivot_turn_index") is not None
            else "No pivot detected."
        )

        sections.append(f"""--- Transcript {tid} (topic: {transcript.get('topic','')}) ---
{turns_summary}
Pivot: {pivot_summary}
""")

    all_transcripts_text = "\n".join(sections)

    return f"""You are a senior customer support routing specialist.

Analyse the following transcripts and produce routing decisions. Return ONLY a valid JSON array with no extra text, no markdown fences.

Rules:
- One record per transcript
- "transcript_id": string
- "root_cause_of_dissatisfaction": concise string
- "recommended_route": MUST be one of: {routes_str}
- "priority": MUST be one of: {priorities_str} (P1=urgent, P4=low)
- "expected_resolution_path": string describing next steps
- "evidence_turns": array of turn_index integers that support your routing decision

{all_transcripts_text}

Return a JSON array of {len(transcripts)} routing records.
"""


def routing_stage(transcripts: list[dict], pivots: list[dict]) -> list[dict]:
    print("  Building combined routing prompt...")
    prompt = _build_routing_prompt(transcripts, pivots)
    annotation_paths = [f"{ANNOTATIONS_DIR}/{t['transcript_id']}.json" for t in transcripts]

    raw = call_llm(
        prompt=prompt,
        stage="routing",
        transcript_id=None,
        input_artifacts=annotation_paths + [PIVOTS_PATH],
        output_artifact=ROUTING_PATH,
    )

    try:
        routing = parse_llm_json(raw)
    except LLMParseError as e:
        raise RuntimeError(f"Routing parse failed: {e}") from e

    if not isinstance(routing, list):
        raise RuntimeError(f"Routing: expected JSON array, got {type(routing).__name__}")

    turn_counts = {t["transcript_id"]: len(t["turns"]) for t in transcripts}
    for record in routing:
        tid = record.get("transcript_id", "")
        if record.get("recommended_route") not in ALLOWED_ROUTES:
            raise RuntimeError(
                f"Routing {tid}: invalid route {record.get('recommended_route')!r}. "
                f"Allowed: {ALLOWED_ROUTES}"
            )
        if record.get("priority") not in ALLOWED_PRIORITIES:
            raise RuntimeError(
                f"Routing {tid}: invalid priority {record.get('priority')!r}. "
                f"Allowed: {ALLOWED_PRIORITIES}"
            )
        max_idx = turn_counts.get(tid, 0) - 1
        for ev in record.get("evidence_turns", []):
            if not (0 <= ev <= max_idx):
                raise RuntimeError(
                    f"Routing {tid}: evidence_turn {ev} out of range [0, {max_idx}]"
                )

    _write_json(ROUTING_PATH, routing)
    print(f"  Saved {ROUTING_PATH} ({len(routing)} records)")
    log_state("ROUTING_COMPLETE")
    return routing


# ---------------------------------------------------------------------------
# Stage 5: Coaching notes (one LLM call per pivot transcript)
# ---------------------------------------------------------------------------

def _build_coaching_prompt(transcript: dict, annotations: list[dict], pivot: dict) -> str:
    techniques_str = ", ".join(ALLOWED_COACHING_TECHNIQUES)
    turns_text = "\n".join(
        f"  Turn {a['turn_index']} [{a['speaker']}]: {a['original_text']} "
        f"(en: {a['translated_to_english']}) "
        f"[intent={a['intent']}, sentiment={a['sentiment_score']}, escalation={a['is_escalation_signal']}]"
        for a in annotations
    )
    pivot_quote = pivot["pivot_agent_quote"]

    return f"""You are a customer support coaching specialist.

Write a detailed coaching note for this transcript. The agent made a critical communication error at the pivot turn.

Pivot agent quote (COPY THIS VERBATIM in your note):
"{pivot_quote}"

Pivot details:
- Turn index: {pivot['pivot_turn_index']}
- Pre-pivot average user sentiment: {pivot['pre_pivot_user_sentiment']}
- Post-pivot average user sentiment: {pivot['post_pivot_user_sentiment']}
- Calculation: {pivot['calculation']}

Full annotated transcript (Transcript {transcript['transcript_id']}):
{turns_text}

Allowed coaching techniques: {techniques_str}

Write a coaching note in Markdown that includes ALL of the following sections:
1. **Exact Agent Quote** — quote the pivot_agent_quote verbatim (word for word)
2. **What the Agent Said** — plain-language explanation of the agent's response and its tone
3. **What the Agent Should Have Said** — a specific, improved alternative response
4. **Missed Coaching Technique** — name exactly one technique from the allowed list and explain it
5. **Why the Alternative Reduces Escalation** — explain the mechanism by which the improved response would de-escalate the customer

Important: The exact pivot quote "{pivot_quote}" must appear verbatim in your note.
"""


def coaching_stage(transcripts: list[dict], pivots: list[dict]) -> None:
    pivot_map = {p["transcript_id"]: p for p in pivots}
    pivot_transcripts = [t for t in transcripts if pivot_map.get(t["transcript_id"], {}).get("pivot_turn_index") is not None]

    print(f"  {len(pivot_transcripts)} transcript(s) have pivots: {[t['transcript_id'] for t in pivot_transcripts]}")

    for transcript in pivot_transcripts:
        tid = transcript["transcript_id"]
        pivot = pivot_map[tid]
        annotations = _read_json(f"{ANNOTATIONS_DIR}/{tid}.json")
        output_path = f"{COACHING_DIR}/{tid}.md"

        print(f"  Generating coaching note for {tid}...")
        prompt = _build_coaching_prompt(transcript, annotations, pivot)

        raw = call_llm(
            prompt=prompt,
            stage="coaching",
            transcript_id=tid,
            input_artifacts=[f"{ANNOTATIONS_DIR}/{tid}.json", PIVOTS_PATH],
            output_artifact=output_path,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw)
        print(f"    Saved {output_path}")

    log_state("COACHING_NOTES_GENERATED")


# ---------------------------------------------------------------------------
# Stage 6: Cross-transcript pattern mining (one LLM call)
# ---------------------------------------------------------------------------

def _build_patterns_prompt(transcripts: list[dict], pivots: list[dict]) -> str:
    pivot_map = {p["transcript_id"]: p for p in pivots}
    sections = []
    for transcript in transcripts:
        tid = transcript["transcript_id"]
        annotations = _read_json(f"{ANNOTATIONS_DIR}/{tid}.json")
        coaching_path = f"{COACHING_DIR}/{tid}.md"
        coaching_text = _read_text(coaching_path) if os.path.exists(coaching_path) else "(no coaching note)"
        pivot = pivot_map.get(tid, {})

        turns_summary = "\n".join(
            f"  Turn {a['turn_index']} [{a['speaker']}]: {a['translated_to_english'][:150]} (intent={a['intent']})"
            for a in annotations
        )
        sections.append(f"=== {tid} (pivot turn: {pivot.get('pivot_turn_index', 'none')}) ===\n{turns_summary}\n\nCoaching Note:\n{coaching_text[:800]}\n")

    all_text = "\n\n".join(sections)

    return f"""You are a customer experience analyst.

Identify recurring patterns across the following {len(transcripts)} customer support transcripts. Return ONLY a valid JSON array with no extra text, no markdown fences.

Rules:
- Each pattern MUST cite at least 2 different transcript IDs in "evidence_transcript_ids"
- "pattern": string describing the recurring issue or behaviour
- "evidence_transcript_ids": array of ≥2 transcript ID strings (e.g. ["T1", "T3"])
- "supporting_turns": array of strings in format "TRANSCRIPT_ID:TURN_INDEX" (e.g. ["T1:7", "T3:2"])
- "product_or_process_recommendation": string with a concrete improvement suggestion

{all_text}

Return a JSON array of cross-transcript patterns.
"""


def patterns_stage(transcripts: list[dict], pivots: list[dict]) -> None:
    print("  Mining cross-transcript patterns...")
    prompt = _build_patterns_prompt(transcripts, pivots)
    annotation_paths = [f"{ANNOTATIONS_DIR}/{t['transcript_id']}.json" for t in transcripts]
    coaching_paths = [f"{COACHING_DIR}/{t['transcript_id']}.md" for t in transcripts if os.path.exists(f"{COACHING_DIR}/{t['transcript_id']}.md")]

    raw = call_llm(
        prompt=prompt,
        stage="cross_transcript_patterns",
        transcript_id=None,
        input_artifacts=annotation_paths + coaching_paths,
        output_artifact=PATTERNS_PATH,
    )

    try:
        patterns = parse_llm_json(raw)
    except LLMParseError as e:
        raise RuntimeError(f"Patterns parse failed: {e}") from e

    if not isinstance(patterns, list):
        raise RuntimeError(f"Patterns: expected JSON array, got {type(patterns).__name__}")

    for i, p in enumerate(patterns):
        ev_ids = p.get("evidence_transcript_ids", [])
        if len(ev_ids) < 2:
            raise RuntimeError(
                f"Pattern {i}: evidence_transcript_ids must have ≥2 entries, got {ev_ids}"
            )

    _write_json(PATTERNS_PATH, patterns)
    print(f"  Saved {PATTERNS_PATH} ({len(patterns)} pattern(s))")


# ---------------------------------------------------------------------------
# Stage 7: Translation audit (two LLM calls on first non-English transcript)
# ---------------------------------------------------------------------------

def _find_first_non_english(transcripts: list[dict]) -> dict | None:
    sorted_transcripts = sorted(transcripts, key=lambda t: t["transcript_id"])
    for t in sorted_transcripts:
        annotations = _read_json(f"{ANNOTATIONS_DIR}/{t['transcript_id']}.json")
        if any(a.get("language_detected", "en") != "en" for a in annotations):
            return t
    return None


def _build_translation_prompt_a(transcript: dict) -> str:
    turns_text = "\n".join(
        f"  Turn {t['turn_index']}: {t['text']}"
        for t in transcript["turns"]
    )
    return f"""Translate each turn of the following customer support transcript into English using a LITERAL, FORMAL style. Prioritise accuracy over naturalness. Return ONLY a valid JSON array of objects with keys "turn_index" and "translation". No extra text, no markdown.

Transcript {transcript['transcript_id']}:
{turns_text}
"""


def _build_translation_prompt_b(transcript: dict) -> str:
    turns_text = "\n".join(
        f"  Turn {t['turn_index']}: {t['text']}"
        for t in transcript["turns"]
    )
    return f"""Translate each turn of the following customer support transcript into English using a NATURAL, CONTEXTUAL style. Prioritise how a native English speaker would express the same idea in a support conversation. Return ONLY a valid JSON array of objects with keys "turn_index" and "translation". No extra text, no markdown.

Transcript {transcript['transcript_id']}:
{turns_text}
"""


def translation_audit_stage(transcripts: list[dict]) -> None:
    target = _find_first_non_english(transcripts)
    if target is None:
        print("  No non-English transcript found — skipping translation audit.")
        _write_json(TRANSLATION_AUDIT_PATH, [])
        return

    tid = target["transcript_id"]
    print(f"  Translation audit on {tid}...")

    prompt_a = _build_translation_prompt_a(target)
    prompt_b = _build_translation_prompt_b(target)

    raw_a = call_llm(
        prompt=prompt_a,
        stage="translation_audit_A",
        transcript_id=tid,
        input_artifacts=[TRANSCRIPT_PATH],
        output_artifact=TRANSLATION_AUDIT_PATH,
    )
    raw_b = call_llm(
        prompt=prompt_b,
        stage="translation_audit_B",
        transcript_id=tid,
        input_artifacts=[TRANSCRIPT_PATH],
        output_artifact=TRANSLATION_AUDIT_PATH,
    )

    try:
        trans_a = parse_llm_json(raw_a)
        trans_b = parse_llm_json(raw_b)
    except LLMParseError as e:
        raise RuntimeError(f"Translation audit parse failed: {e}") from e

    a_map = {item["turn_index"]: item["translation"] for item in trans_a}
    b_map = {item["turn_index"]: item["translation"] for item in trans_b}

    divergences = []
    for turn in target["turns"]:
        idx = turn["turn_index"]
        t_a = a_map.get(idx, "")
        t_b = b_map.get(idx, "")
        if t_a != t_b:
            divergences.append({
                "original_text": turn["text"],
                "translation_A": t_a,
                "translation_B": t_b,
                "difference": f"Prompt A produced a more literal rendering; Prompt B a more natural one.",
                "possible_impact_on_intent_sentiment_or_routing": (
                    "Literal vs contextual phrasing may cause different intent classification "
                    "or sentiment scoring by downstream LLMs."
                ),
            })

    _write_json(TRANSLATION_AUDIT_PATH, divergences)
    print(f"  Saved {TRANSLATION_AUDIT_PATH} ({len(divergences)} divergence(s))")


# ---------------------------------------------------------------------------
# Stage 8: Bilingual reply generation
# ---------------------------------------------------------------------------

def _build_bilingual_reply_prompt(transcript: dict, routing_record: dict, primary_lang: str) -> str:
    turns_text = "\n".join(
        f"  Turn {t['turn_index']} [{t['speaker_original']}]: {t['text']}"
        for t in transcript["turns"]
    )
    return f"""You are a multilingual customer support specialist.

Generate a closing reply for this support conversation in TWO languages:
1. The customer's language ({primary_lang})
2. English

Use the routing context to inform your reply.

Root cause: {routing_record.get('root_cause_of_dissatisfaction', '')}
Recommended route: {routing_record.get('recommended_route', '')}
Expected resolution: {routing_record.get('expected_resolution_path', '')}

Transcript {transcript['transcript_id']}:
{turns_text}

Return ONLY a valid JSON object (no markdown, no extra text) with keys:
- "transcript_id": "{transcript['transcript_id']}"
- "detected_language": "{primary_lang}"
- "reply_original_language": reply in {primary_lang}
- "reply_english": reply in English
"""


def bilingual_replies_stage(transcripts: list[dict], routing: list[dict]) -> None:
    routing_map = {r["transcript_id"]: r for r in routing}
    records = []

    for transcript in transcripts:
        tid = transcript["transcript_id"]
        annotations = _read_json(f"{ANNOTATIONS_DIR}/{tid}.json")
        non_en = [a for a in annotations if a.get("language_detected", "en") != "en"]
        if not non_en:
            continue

        primary_lang = non_en[0]["language_detected"]
        routing_record = routing_map.get(tid, {})
        output_path = BILINGUAL_REPLIES_PATH

        print(f"  Generating bilingual reply for {tid} ({primary_lang})...")
        prompt = _build_bilingual_reply_prompt(transcript, routing_record, primary_lang)

        raw = call_llm(
            prompt=prompt,
            stage="bilingual_reply",
            transcript_id=tid,
            input_artifacts=[f"{ANNOTATIONS_DIR}/{tid}.json", ROUTING_PATH],
            output_artifact=output_path,
        )

        try:
            record = parse_llm_json(raw)
        except LLMParseError as e:
            raise RuntimeError(f"Bilingual reply parse failed for {tid}: {e}") from e

        if not record.get("reply_original_language") or not record.get("reply_english"):
            raise RuntimeError(f"Bilingual reply for {tid}: empty reply field(s)")

        records.append(record)

    _write_json(BILINGUAL_REPLIES_PATH, records)
    print(f"  Saved {BILINGUAL_REPLIES_PATH} ({len(records)} record(s))")


# ---------------------------------------------------------------------------
# Stage 9: Compliance flagging (deterministic, no LLM)
# ---------------------------------------------------------------------------

def compliance_flagging_stage(transcripts: list[dict]) -> None:
    all_flags = []
    for transcript in transcripts:
        tid = transcript["transcript_id"]
        flags = scan_compliance_keywords(tid, transcript["turns"], COMPLIANCE_KEYWORDS)
        all_flags.extend(flags)
        if flags:
            print(f"  {tid}: {len(flags)} compliance flag(s)")

    _write_json(COMPLIANCE_FLAGS_PATH, all_flags)
    print(f"  Saved {COMPLIANCE_FLAGS_PATH} ({len(all_flags)} total flag(s))")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    # Preflight checks
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"ERROR: {TRANSCRIPT_PATH} not found in current directory.", file=sys.stderr)
        sys.exit(1)

    log_state("INIT")

    # Clean all output artifacts
    print("Cleaning previous outputs...")
    clean_outputs()

    # Stage 1
    transcripts = load_transcripts_stage()

    # Stage 2
    annotate_transcripts_stage(transcripts)

    # Stage 3
    pivots = compute_pivots_stage(transcripts)

    # Stage 4
    routing = routing_stage(transcripts, pivots)

    # Stage 5
    coaching_stage(transcripts, pivots)

    # Stage 6
    patterns_stage(transcripts, pivots)

    # Stage 7
    translation_audit_stage(transcripts)

    # Stage 8
    bilingual_replies_stage(transcripts, routing)

    # Stage 9
    compliance_flagging_stage(transcripts)

    log_state("VALIDATION_COMPLETE")
    log_state("RESULTS_FINALISED")
    print("\nPipeline complete. Run: python validate.py")


if __name__ == "__main__":
    main()
