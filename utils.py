import json
import re

from vocab import ALLOWED_INTENTS, COMPLIANCE_KEYWORDS, LATIN_SCRIPT_LANGS, SPEAKER_NORMALISATION, is_agent, is_user

TRANSCRIPT_PATH = "transcript.json"

REQUIRED_ANNOTATION_FIELDS = {
    "turn_index",
    "speaker",
    "original_text",
    "language_detected",
    "translated_to_english",
    "intent",
    "sentiment_score",
    "is_escalation_signal",
    "unmet_information_need",
}


class LLMParseError(Exception):
    def __init__(self, raw: str):
        self.raw = raw
        super().__init__(f"Failed to parse LLM JSON. Raw response (first 500 chars): {raw[:500]}")


def load_transcripts(path: str = TRANSCRIPT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array at root")

    transcripts = []
    for t in data:
        if "transcript_id" not in t:
            raise ValueError(f"Transcript missing 'transcript_id': {str(t)[:100]}")
        if "turns" not in t or not isinstance(t["turns"], list):
            raise ValueError(f"Transcript {t.get('transcript_id')!r} missing 'turns' array")

        turns = []
        for idx, turn in enumerate(t["turns"]):
            raw_speaker = turn.get("speaker", "")
            if raw_speaker not in SPEAKER_NORMALISATION:
                raise ValueError(
                    f"Transcript {t['transcript_id']!r} turn {idx}: "
                    f"unknown speaker label {raw_speaker!r}. "
                    f"Add to SPEAKER_NORMALISATION in vocab.py."
                )
            turns.append({
                **turn,
                "turn_index": idx,
                "speaker": SPEAKER_NORMALISATION[raw_speaker],
                "speaker_original": raw_speaker,
            })

        transcripts.append({**t, "turns": turns})

    return transcripts


def parse_llm_json(raw: str):
    stripped = raw.strip()
    # Pass 1: direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Pass 2: extract first JSON array or object (handles markdown code fences etc.)
    match = re.search(r"(\[.*\]|\{.*\})", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise LLMParseError(raw)


def validate_annotation(ann: dict, allowed_intents: list[str] = ALLOWED_INTENTS) -> None:
    missing = REQUIRED_ANNOTATION_FIELDS - set(ann.keys())
    if missing:
        raise ValueError(f"Annotation missing fields: {missing}. Got: {list(ann.keys())}")

    if ann["intent"] not in allowed_intents:
        raise ValueError(
            f"Invalid intent {ann['intent']!r}. Allowed: {allowed_intents}"
        )

    score = ann["sentiment_score"]
    if not isinstance(score, (int, float)):
        raise ValueError(f"sentiment_score must be numeric, got {type(score).__name__}: {score!r}")
    if not (-1.0 <= float(score) <= 1.0):
        raise ValueError(f"sentiment_score {score} out of range [-1.0, 1.0]")


def compute_pivot(transcript_id: str, annotations: list[dict]) -> dict:
    user_turns = [a for a in annotations if is_user(a["speaker"])]
    agent_turns = [a for a in annotations if is_agent(a["speaker"])]

    print(f"[PIVOT DEBUG] {transcript_id}")
    print(f"  User turn indexes:  {[u['turn_index'] for u in user_turns]}")
    print(f"  Agent turn indexes: {[a['turn_index'] for a in agent_turns]}")

    null_record = {
        "transcript_id": transcript_id,
        "pivot_turn_index": None,
        "pivot_agent_quote": None,
        "pre_pivot_user_sentiment": None,
        "post_pivot_user_sentiment": None,
        "calculation": "",
    }

    if not user_turns:
        return {**null_record, "calculation": "No pivot detected because there are no user turns."}
    if not agent_turns:
        return {**null_record, "calculation": "No pivot detected because there are no agent turns."}

    for agent_turn in agent_turns:
        t = agent_turn["turn_index"]
        prior_user = [u for u in user_turns if u["turn_index"] < t]
        subsequent_user = [u for u in user_turns if u["turn_index"] > t]

        if not prior_user:
            print(f"  Agent turn {t}: skipped (no prior user turns)")
            continue
        if not subsequent_user:
            print(f"  Agent turn {t}: skipped (no subsequent user turns)")
            continue

        pre_avg = sum(u["sentiment_score"] for u in prior_user) / len(prior_user)
        post_avg = sum(u["sentiment_score"] for u in subsequent_user) / len(subsequent_user)
        delta = post_avg - pre_avg
        qualifies = delta <= -0.25 or post_avg <= -0.60

        print(f"  Agent turn {t}: pre_avg={pre_avg:.3f}, post_avg={post_avg:.3f}, delta={delta:.3f}, pivot={qualifies}")

        if qualifies:
            calc = (
                f"Agent turn at index {t}. "
                f"Pre-avg user sentiment: {pre_avg:.3f} (over {len(prior_user)} turn(s)). "
                f"Post-avg user sentiment: {post_avg:.3f} (over {len(subsequent_user)} turn(s)). "
                f"Delta: {delta:.3f}. "
                f"Pivot criteria met: delta <= -0.25 is {delta <= -0.25}, "
                f"post_avg <= -0.60 is {post_avg <= -0.60}."
            )
            return {
                "transcript_id": transcript_id,
                "pivot_turn_index": t,
                "pivot_agent_quote": agent_turn["original_text"],
                "pre_pivot_user_sentiment": round(pre_avg, 4),
                "post_pivot_user_sentiment": round(post_avg, 4),
                "calculation": calc,
            }

    return {
        **null_record,
        "calculation": (
            "No pivot detected because no agent turn produced a user sentiment "
            "delta <= -0.25 or post-pivot average <= -0.60."
        ),
    }


def scan_compliance_keywords(
    transcript_id: str,
    turns: list[dict],
    keywords: dict = COMPLIANCE_KEYWORDS,
) -> list[dict]:
    flags = []
    for turn in turns:
        text = turn.get("text", "") or turn.get("original_text", "")
        for phrase, (lang, escalation) in keywords.items():
            if lang in LATIN_SCRIPT_LANGS:
                match = phrase.lower() in text.lower()
            else:
                match = phrase in text

            if match:
                flags.append({
                    "transcript_id": transcript_id,
                    "turn_index": turn.get("turn_index"),
                    "trigger_phrase": phrase,
                    "language": lang,
                    "recommended_escalation": escalation,
                    "reason": (
                        f"Trigger phrase {phrase!r} detected in turn {turn.get('turn_index')} "
                        f"of transcript {transcript_id}."
                    ),
                })
    return flags
