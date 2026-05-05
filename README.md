# DerivSense

**DerivSense** is a replayable, staged Python pipeline for multilingual customer-support transcript analysis. It reads raw support-chat transcripts, annotates every turn using an LLM via OpenRouter, detects the agent's worst communication pivot using deterministic Python, generates routing decisions, produces quote-anchored coaching notes, and validates every output — all from a single command.

---

## Overview

Customer support quality analysis is hard to scale across languages. DerivSense automates it end-to-end:

- **Annotates** every turn with intent, sentiment score, escalation signal, and English translation
- **Detects pivots** — the exact agent response that caused the biggest sentiment crash — using pure Python math, no LLM guessing
- **Routes** each transcript to the right team (payments, compliance, risk, legal, etc.) with priority and evidence
- **Coaches** agents with verbatim-quoted feedback anchored to the worst moment in the conversation
- **Flags compliance risks** using deterministic multilingual keyword rules
- **Validates** every artifact for schema correctness, vocabulary conformance, and cross-artifact integrity

Supports **English, Spanish, Bahasa Indonesia, Mandarin, and Arabic** out of the box.

---

## Transcripts

Input file: `transcript.json` — an array of support chat transcripts.

| ID | Language | Topic |
|----|----------|-------|
| T1 | English | Deposit failure (Skrill, 200 USD) |
| T2 | Spanish | Account verification / document rejection |
| T3 | Bahasa Indonesia | Forced margin call / fraud accusation |
| T4 | Mandarin | KYC dispute / frozen account |
| T5 | Arabic | Bonus dispute / missing deposit bonus |

Schema per transcript:
```json
{
  "transcript_id": "T1",
  "language_hint": "English",
  "topic": "deposit failure",
  "turns": [
    { "speaker": "User", "text": "hi my deposit of 200 USD..." },
    { "speaker": "Agent", "text": "Hello! Can you please share..." }
  ]
}
```

Speaker labels are native-language (e.g. `Usuario`, `Pengguna`, `用户`, `المستخدم`) and are normalised automatically by the pipeline.

---

## Quick Start

```bash
# 1. Set your OpenRouter API key
export OPENROUTER_API_KEY=your_key_here

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Validate all outputs
python validate.py
```

Python 3.10+ required.

---

## Pipeline Stages

```
INIT
  └─ Load transcript.json, inject turn_index, normalise speaker labels

TRANSCRIPTS_LOADED
  └─ 5 transcripts ready (T1–T5)

TURN_ANNOTATIONS_COMPLETE
  └─ 1 LLM call per transcript → annotations/{id}.json
     Each turn: intent, sentiment_score, language_detected,
     translated_to_english, is_escalation_signal, unmet_information_need

SENTIMENT_TRAJECTORIES_COMPUTED → PIVOTS_DETECTED
  └─ Pure Python — no LLM
     For each agent turn: pre_avg / post_avg user sentiment computed
     Pivot = agent turn with the largest sentiment delta (most negative)
     Criteria: delta ≤ −0.25 OR post_avg ≤ −0.60
     Result → pivots.json

ROUTING_COMPLETE
  └─ 1 combined LLM call (all annotations + pivots)
     Produces root cause, route, priority, evidence turns → routing.json

COACHING_NOTES_GENERATED
  └─ 1 LLM call per pivot transcript
     Verbatim pivot quote + what was said / should have been said
     Missed coaching technique + why alternative reduces escalation
     → coaching/{id}.md

CROSS-TRANSCRIPT PATTERNS
  └─ 1 LLM call → cross_transcript_patterns.json

TRANSLATION AUDIT
  └─ 2 LLM calls (literal vs contextual) on T2 (Spanish)
     Divergences → translation_audit.json

BILINGUAL REPLIES
  └─ 1 LLM call per non-English transcript (T2–T5)
     Reply in original language + English → bilingual_replies.json

COMPLIANCE FLAGGING
  └─ Deterministic keyword scan — no LLM
     Multilingual triggers → compliance_flags.json

RESULTS_FINALISED
```

---

## Output Artifacts

| File / Directory | Description |
|---|---|
| `annotations/T1.json` … `T5.json` | Per-turn LLM annotations for every transcript |
| `pivots.json` | Strongest-delta pivot per transcript: turn index, verbatim agent quote, pre/post sentiment, calculation |
| `routing.json` | Root cause, recommended route, priority (P1–P4), resolution path, evidence turn indexes |
| `coaching/T1.md` … `T5.md` | Agent coaching notes anchored to exact pivot quote |
| `cross_transcript_patterns.json` | Recurring patterns across ≥2 transcripts with process recommendations |
| `translation_audit.json` | Translation divergences (literal vs contextual) on T2 with impact assessment |
| `bilingual_replies.json` | Closing replies in original language + English for T2–T5 |
| `compliance_flags.json` | Multilingual keyword flags with recommended escalation (compliance / legal / risk) |
| `llm_calls.jsonl` | Append-only log: stage, transcript_id, model, prompt hash, input/output artifacts, timestamp |

---

## Pivot Detection Algorithm

Pivot detection is fully deterministic — the LLM only supplies `sentiment_score` values; the decision logic is pure Python.

```
For each agent turn T (skipping first/last if no prior/subsequent user turns):
    pre_avg  = mean(sentiment of user turns before T)
    post_avg = mean(sentiment of user turns after T)
    delta    = post_avg - pre_avg

    qualifies if: delta ≤ -0.25  (material drop)
               OR post_avg ≤ -0.60  (remains strongly negative)

Select: agent turn with the most negative delta (strongest sentiment crash)
```

Debug output is printed for every evaluated agent turn:
```
[PIVOT DEBUG] T1
  User turn indexes:  [0, 2, 4, 6, 8]
  Agent turn indexes: [1, 3, 5, 7, 9]
  Agent turn 7: pre_avg=0.050, post_avg=-0.700, delta=-0.750, pivot=True
```

---

## Controlled Vocabularies

All LLM outputs are validated against these before any artifact is written:

**Intents:** `deposit_issue`, `withdrawal_issue`, `kyc_issue`, `margin_dispute`, `bonus_dispute`, `account_access`, `technical_issue`, `complaint`, `information_request`, `identity_verification`, `escalation_request`, `other`

**Routes:** `auto_resolve`, `payments`, `compliance`, `risk`, `retention`, `legal`

**Priorities:** `P1` (urgent) → `P4` (low)

**Coaching Techniques:** `expectation_setting`, `ownership_language`, `proactive_disclosure`, `empathy_with_resolution`, `clear_next_step`, `avoid_repetition`, `complaint_handling`, `policy_explanation`

---

## Compliance Keyword Coverage

Deterministic scan — no LLM — across all 5 languages:

| Language | Sample triggers |
|----------|----------------|
| English | complaint, fraud, scam, frozen account, withdraw, legal |
| Spanish | queja, fraude, retirar |
| Bahasa Indonesia | penipuan, rugi, paksa |
| Mandarin | 投诉, 冻结, 欺诈 |
| Arabic | احتيال, شكوى, سحب |

---

## Validation

`validate.py` runs 7 check groups and exits `0` (all pass) or `1` (any fail):

1. **File existence** — all required files and directories present
2. **JSON validity** — every `.json` file and every `llm_calls.jsonl` line parses cleanly
3. **Per-transcript completeness** — annotation count matches turn count; `original_text` matches source verbatim
4. **Vocabulary conformance** — all intents, routes, priorities within allowed values
5. **Pivot integrity** — pivot quote matches annotation; quote appears verbatim in coaching note; false-negative detection (pivot cannot say "no user turns" if user turns exist)
6. **Routing evidence** — all `evidence_turns` indexes reference real turns
7. **LLM log completeness** — required stage entries present for annotation, routing, coaching, patterns

---

## File Structure

```
DerivSense/
├── main.py               # Pipeline orchestration
├── validate.py           # Post-run validation
├── llm_client.py         # OpenRouter API client + JSONL logger
├── utils.py              # load_transcripts, compute_pivot, scan_compliance_keywords, parse_llm_json
├── vocab.py              # Controlled vocabularies + speaker normalisation helpers
├── transcript.json       # Input: 5 multilingual support transcripts
├── requirements.txt
│
├── annotations/          # Per-transcript LLM turn annotations
├── coaching/             # Agent coaching notes (Markdown)
├── pivots.json
├── routing.json
├── cross_transcript_patterns.json
├── translation_audit.json
├── bilingual_replies.json
├── compliance_flags.json
└── llm_calls.jsonl
```

---

## LLM Usage

- **Model:** `openai/gpt-4o-mini` via [OpenRouter](https://openrouter.ai)
- **Calls per full run:** `N` annotation + 1 routing + `P` coaching + 1 patterns + 2 translation audit + `M` bilingual replies
  - For the included 5-transcript dataset: ~14 LLM calls total
- Every call is logged to `llm_calls.jsonl` with a SHA-256 prompt hash for auditability

---

## Notes

- The pipeline deletes all previous output artifacts at the start of each run — fully replayable from `transcript.json`.
- Routing and coaching are gated: they cannot run before annotation and pivot detection complete.
- Compliance flagging has zero LLM cost — pure keyword matching.
- Replace `transcript.json` with any file following the same schema to analyse a new dataset.
