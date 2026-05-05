import hashlib
import json
import os
from datetime import datetime, timezone

import requests

LLM_LOG_PATH = "llm_calls.jsonl"


class LLMError(Exception):
    pass


def call_llm(
    prompt: str,
    stage: str,
    transcript_id: str | None,
    input_artifacts: list[str],
    output_artifact: str,
    model: str = "openai/gpt-4o-mini",
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not set")

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise LLMError(
            f"OpenRouter HTTP {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    choices = data.get("choices")
    if not choices:
        raise LLMError(f"No choices in response: {json.dumps(data)[:500]}")

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise LLMError("Empty content in LLM response")

    log_entry = {
        "stage": stage,
        "transcript_id": transcript_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openrouter",
        "model": model,
        "prompt_hash": prompt_hash,
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
    }
    with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return content
