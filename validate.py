"""
Post-run validation script.

Run:
    python validate.py

Exits 0 if all checks pass, 1 if any fail.
"""

import json
import os
import sys

from vocab import ALLOWED_INTENTS, ALLOWED_PRIORITIES, ALLOWED_ROUTES, is_user

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


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Validator:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  PASS  {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        self.errors.append(msg)
        print(f"  FAIL  {msg}")

    def check(self, condition: bool, pass_msg: str, fail_msg: str) -> bool:
        if condition:
            self.ok(pass_msg)
        else:
            self.fail(fail_msg)
        return condition


# ---------------------------------------------------------------------------
# Check 1: File existence
# ---------------------------------------------------------------------------

def check_file_existence(v: Validator) -> None:
    print("\n[1] File and directory existence")
    required_files = [
        TRANSCRIPT_PATH,
        PIVOTS_PATH,
        ROUTING_PATH,
        LLM_LOG_PATH,
        PATTERNS_PATH,
        TRANSLATION_AUDIT_PATH,
        BILINGUAL_REPLIES_PATH,
        COMPLIANCE_FLAGS_PATH,
    ]
    required_dirs = [ANNOTATIONS_DIR, COACHING_DIR]

    for path in required_files:
        v.check(os.path.exists(path), f"{path} exists", f"{path} is MISSING")

    for d in required_dirs:
        v.check(os.path.isdir(d), f"{d}/ directory exists", f"{d}/ directory is MISSING")


# ---------------------------------------------------------------------------
# Check 2: JSON validity
# ---------------------------------------------------------------------------

def check_json_validity(v: Validator, transcripts: list[dict]) -> None:
    print("\n[2] JSON validity")
    json_files = [
        PIVOTS_PATH,
        ROUTING_PATH,
        PATTERNS_PATH,
        TRANSLATION_AUDIT_PATH,
        BILINGUAL_REPLIES_PATH,
        COMPLIANCE_FLAGS_PATH,
    ]
    for t in transcripts:
        json_files.append(f"{ANNOTATIONS_DIR}/{t['transcript_id']}.json")

    for path in json_files:
        if not os.path.exists(path):
            v.fail(f"{path}: file missing — cannot validate JSON")
            continue
        try:
            _load_json(path)
            v.ok(f"{path}: valid JSON")
        except json.JSONDecodeError as e:
            v.fail(f"{path}: JSON parse error — {e}")

    if os.path.exists(LLM_LOG_PATH):
        with open(LLM_LOG_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    v.fail(f"{LLM_LOG_PATH} line {i}: JSON parse error — {e}")
        v.ok(f"{LLM_LOG_PATH}: all lines valid JSON")
    else:
        v.fail(f"{LLM_LOG_PATH}: file missing")


# ---------------------------------------------------------------------------
# Check 3: Per-transcript completeness
# ---------------------------------------------------------------------------

def check_transcript_completeness(v: Validator, transcripts: list[dict]) -> None:
    print("\n[3] Per-transcript completeness")
    for t in transcripts:
        tid = t["transcript_id"]
        expected_turns = len(t["turns"])
        ann_path = f"{ANNOTATIONS_DIR}/{tid}.json"

        if not os.path.exists(ann_path):
            v.fail(f"{tid}: annotation file {ann_path} missing")
            continue

        try:
            annotations = _load_json(ann_path)
        except json.JSONDecodeError:
            v.fail(f"{tid}: annotation file invalid JSON")
            continue

        v.check(
            len(annotations) == expected_turns,
            f"{tid}: annotation count {len(annotations)} == {expected_turns}",
            f"{tid}: annotation count mismatch — expected {expected_turns}, got {len(annotations)}",
        )

        source_texts = {turn["turn_index"]: turn["text"] for turn in t["turns"]}
        for ann in annotations:
            idx = ann.get("turn_index")
            if idx not in source_texts:
                v.fail(f"{tid} turn {idx}: turn_index not in source transcript")
                continue
            if ann.get("original_text") != source_texts[idx]:
                v.fail(
                    f"{tid} turn {idx}: original_text mismatch\n"
                    f"    source:     {source_texts[idx]!r}\n"
                    f"    annotation: {ann.get('original_text')!r}"
                )
            else:
                v.ok(f"{tid} turn {idx}: original_text matches source")


# ---------------------------------------------------------------------------
# Check 4: Vocabulary conformance
# ---------------------------------------------------------------------------

def check_vocab_conformance(v: Validator, transcripts: list[dict]) -> None:
    print("\n[4] Vocabulary conformance")
    for t in transcripts:
        tid = t["transcript_id"]
        ann_path = f"{ANNOTATIONS_DIR}/{tid}.json"
        if not os.path.exists(ann_path):
            continue
        try:
            annotations = _load_json(ann_path)
        except json.JSONDecodeError:
            continue
        for ann in annotations:
            intent = ann.get("intent")
            v.check(
                intent in ALLOWED_INTENTS,
                f"{tid} turn {ann.get('turn_index')}: intent {intent!r} valid",
                f"{tid} turn {ann.get('turn_index')}: invalid intent {intent!r}",
            )
            score = ann.get("sentiment_score")
            v.check(
                isinstance(score, (int, float)) and -1.0 <= float(score) <= 1.0,
                f"{tid} turn {ann.get('turn_index')}: sentiment_score {score} in range",
                f"{tid} turn {ann.get('turn_index')}: invalid sentiment_score {score!r}",
            )

    if os.path.exists(ROUTING_PATH):
        try:
            routing = _load_json(ROUTING_PATH)
        except json.JSONDecodeError:
            routing = []
        for record in routing:
            tid = record.get("transcript_id", "?")
            v.check(
                record.get("recommended_route") in ALLOWED_ROUTES,
                f"{tid}: route {record.get('recommended_route')!r} valid",
                f"{tid}: invalid route {record.get('recommended_route')!r}",
            )
            v.check(
                record.get("priority") in ALLOWED_PRIORITIES,
                f"{tid}: priority {record.get('priority')!r} valid",
                f"{tid}: invalid priority {record.get('priority')!r}",
            )


# ---------------------------------------------------------------------------
# Check 5: Pivot integrity
# ---------------------------------------------------------------------------

def check_pivot_integrity(v: Validator, transcripts: list[dict]) -> None:
    print("\n[5] Pivot integrity")
    if not os.path.exists(PIVOTS_PATH):
        v.fail(f"{PIVOTS_PATH}: missing — skipping pivot checks")
        return

    try:
        pivots = _load_json(PIVOTS_PATH)
    except json.JSONDecodeError:
        v.fail(f"{PIVOTS_PATH}: invalid JSON")
        return

    pivot_map = {p["transcript_id"]: p for p in pivots}
    for t in transcripts:
        tid = t["transcript_id"]
        pivot = pivot_map.get(tid)
        if not pivot:
            v.fail(f"{tid}: no pivot record in {PIVOTS_PATH}")
            continue

        if pivot.get("pivot_turn_index") is None:
            # False-negative check: if pivot says "no user turns" but annotations
            # actually contain user-speaker turns, that is a speaker-normalisation bug.
            ann_path_check = f"{ANNOTATIONS_DIR}/{tid}.json"
            if os.path.exists(ann_path_check):
                try:
                    anns_check = _load_json(ann_path_check)
                    user_turn_count = sum(1 for a in anns_check if is_user(a.get("speaker", "")))
                    calc = pivot.get("calculation", "")
                    if user_turn_count > 0 and "no user turns" in calc.lower():
                        v.fail(
                            f"{tid}: pivot says 'no user turns' but annotations contain "
                            f"{user_turn_count} user turn(s) — speaker normalisation bug"
                        )
                    else:
                        v.ok(f"{tid}: no pivot (null) — no integrity check needed")
                except json.JSONDecodeError:
                    v.ok(f"{tid}: no pivot (null) — annotation unreadable, skipping false-negative check")
            else:
                v.ok(f"{tid}: no pivot (null) — no integrity check needed")
            continue

        ann_path = f"{ANNOTATIONS_DIR}/{tid}.json"
        if not os.path.exists(ann_path):
            v.fail(f"{tid}: annotation file missing — cannot verify pivot quote")
            continue

        try:
            annotations = _load_json(ann_path)
        except json.JSONDecodeError:
            v.fail(f"{tid}: annotation file invalid JSON")
            continue

        turn_idx = pivot["pivot_turn_index"]
        pivot_quote = pivot.get("pivot_agent_quote", "")
        ann_at_idx = next((a for a in annotations if a.get("turn_index") == turn_idx), None)

        if ann_at_idx is None:
            v.fail(f"{tid}: no annotation at pivot_turn_index {turn_idx}")
        else:
            v.check(
                ann_at_idx.get("original_text") == pivot_quote,
                f"{tid}: pivot_agent_quote matches annotation original_text at turn {turn_idx}",
                f"{tid}: pivot_agent_quote mismatch at turn {turn_idx}\n"
                f"    pivot record: {pivot_quote!r}\n"
                f"    annotation:   {ann_at_idx.get('original_text')!r}",
            )

        coaching_path = f"{COACHING_DIR}/{tid}.md"
        if not os.path.exists(coaching_path):
            v.fail(f"{tid}: coaching file {coaching_path} missing — cannot verify pivot quote inclusion")
        else:
            with open(coaching_path, encoding="utf-8") as cf:
                content = cf.read()
            v.check(
                pivot_quote in content,
                f"{tid}: pivot_agent_quote found verbatim in coaching note",
                f"{tid}: pivot_agent_quote NOT found verbatim in coaching note\n    Quote: {pivot_quote!r}",
            )


# ---------------------------------------------------------------------------
# Check 6: Routing evidence turn indexes
# ---------------------------------------------------------------------------

def check_routing_evidence(v: Validator, transcripts: list[dict]) -> None:
    print("\n[6] Routing evidence turn indexes")
    if not os.path.exists(ROUTING_PATH):
        v.fail(f"{ROUTING_PATH}: missing")
        return

    try:
        routing = _load_json(ROUTING_PATH)
    except json.JSONDecodeError:
        v.fail(f"{ROUTING_PATH}: invalid JSON")
        return

    turn_count_map = {t["transcript_id"]: len(t["turns"]) for t in transcripts}
    for record in routing:
        tid = record.get("transcript_id", "?")
        max_idx = turn_count_map.get(tid, 0) - 1
        for ev in record.get("evidence_turns", []):
            v.check(
                0 <= ev <= max_idx,
                f"{tid}: evidence_turn {ev} valid (max={max_idx})",
                f"{tid}: evidence_turn {ev} out of range [0, {max_idx}]",
            )


# ---------------------------------------------------------------------------
# Check 7: LLM log completeness
# ---------------------------------------------------------------------------

def check_llm_log(v: Validator, transcripts: list[dict], pivots: list[dict]) -> None:
    print("\n[7] LLM log completeness")
    if not os.path.exists(LLM_LOG_PATH):
        v.fail(f"{LLM_LOG_PATH}: missing")
        return

    entries = []
    with open(LLM_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    annotation_tids = {t["transcript_id"] for t in transcripts}
    logged_annotation_tids = {e["transcript_id"] for e in entries if e.get("stage") == "annotation"}
    for tid in annotation_tids:
        v.check(
            tid in logged_annotation_tids,
            f"annotation log entry present for {tid}",
            f"annotation log entry MISSING for {tid}",
        )

    routing_entries = [e for e in entries if e.get("stage") == "routing"]
    v.check(len(routing_entries) >= 1, "routing log entry present", "routing log entry MISSING")

    pivot_tids = {p["transcript_id"] for p in pivots if p.get("pivot_turn_index") is not None}
    logged_coaching_tids = {e["transcript_id"] for e in entries if e.get("stage") == "coaching"}
    for tid in pivot_tids:
        v.check(
            tid in logged_coaching_tids,
            f"coaching log entry present for {tid}",
            f"coaching log entry MISSING for {tid}",
        )

    patterns_entries = [e for e in entries if e.get("stage") == "cross_transcript_patterns"]
    v.check(
        len(patterns_entries) >= 1,
        "cross_transcript_patterns log entry present",
        "cross_transcript_patterns log entry MISSING",
    )

    audit_a = [e for e in entries if e.get("stage") == "translation_audit_A"]
    audit_b = [e for e in entries if e.get("stage") == "translation_audit_B"]
    if audit_a or audit_b:
        v.check(len(audit_a) >= 1, "translation_audit_A log entry present", "translation_audit_A log entry MISSING")
        v.check(len(audit_b) >= 1, "translation_audit_B log entry present", "translation_audit_B log entry MISSING")

    bilingual_entries = [e for e in entries if e.get("stage") == "bilingual_reply"]
    v.ok(f"bilingual_reply log entries: {len(bilingual_entries)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Validation Report — multilingual-transcript-pipeline")
    print("=" * 60)

    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"FATAL: {TRANSCRIPT_PATH} not found. Cannot validate.", file=sys.stderr)
        sys.exit(1)

    with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
        transcripts_raw = json.load(f)

    # Inject turn_index into source transcripts for validation (mirrors load_transcripts)
    transcripts = []
    for t in transcripts_raw:
        turns = [{"turn_index": i, **turn} for i, turn in enumerate(t["turns"])]
        transcripts.append({**t, "turns": turns})

    pivots = []
    if os.path.exists(PIVOTS_PATH):
        try:
            pivots = json.load(open(PIVOTS_PATH, encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    v = Validator()
    check_file_existence(v)
    check_json_validity(v, transcripts)
    check_transcript_completeness(v, transcripts)
    check_vocab_conformance(v, transcripts)
    check_pivot_integrity(v, transcripts)
    check_routing_evidence(v, transcripts)
    check_llm_log(v, transcripts, pivots)

    print("\n" + "=" * 60)
    print(f"Results: {v.passed} passed, {v.failed} failed")
    if v.errors:
        print("\nFailures:")
        for err in v.errors:
            print(f"  - {err}")
    print("=" * 60)

    sys.exit(0 if v.failed == 0 else 1)


if __name__ == "__main__":
    main()
