# Multilingual Transcript Analysis Pipeline

Replayable, staged Python pipeline that analyses multilingual customer-support transcripts using OpenRouter LLMs.

## Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key
- `transcript.json` in the project root (5 transcripts: T1–T5 in EN/ES/ID/ZH/AR)

## Setup and Run

```bash
# 1. Set API key
export OPENROUTER_API_KEY=your_key

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (regenerates all artifacts from scratch)
python main.py

# 4. Validate all outputs
python validate.py
```

## Output Artifacts

| File / Directory | Description |
|---|---|
| `annotations/{transcript_id}.json` | Per-turn LLM annotations (intent, sentiment, escalation signal, translation) |
| `pivots.json` | Deterministic pivot detection per transcript (first agent turn after which user sentiment drops) |
| `routing.json` | Root cause analysis, recommended route, priority, and evidence turns per transcript |
| `coaching/{transcript_id}.md` | Quote-anchored coaching notes for transcripts with detected pivots |
| `cross_transcript_patterns.json` | Recurring patterns across ≥2 transcripts with process recommendations |
| `translation_audit.json` | Turn-by-turn divergences between two translation prompt styles (on first non-English transcript) |
| `bilingual_replies.json` | Closing replies in original language + English for non-English transcripts |
| `compliance_flags.json` | Deterministic keyword-based compliance/legal/risk flags across all turns |
| `llm_calls.jsonl` | Append-only log of every LLM call (stage, model, prompt hash, artifacts) |

## Pipeline Stages

```
INIT
→ TRANSCRIPTS_LOADED          (load transcript.json, inject turn_index, normalise speaker labels)
→ TURN_ANNOTATIONS_COMPLETE   (1 LLM call per transcript)
→ SENTIMENT_TRAJECTORIES_COMPUTED
→ PIVOTS_DETECTED             (pure Python — no LLM)
→ ROUTING_COMPLETE            (1 combined LLM call)
→ COACHING_NOTES_GENERATED    (1 LLM call per pivot transcript)
→ [cross-transcript patterns, translation audit, bilingual replies, compliance flagging]
→ RESULTS_FINALISED
```

## Notes

- The pipeline **deletes all previous outputs** at the start of each run.
- Routing and coaching run only after annotation and pivot detection are complete.
- Compliance flagging is entirely deterministic (no LLM).
- `validate.py` exits 0 on all checks passing, 1 on any failure.
# DerivSense
