ALLOWED_INTENTS = [
    "deposit_issue",
    "withdrawal_issue",
    "kyc_issue",
    "margin_dispute",
    "bonus_dispute",
    "account_access",
    "technical_issue",
    "complaint",
    "information_request",
    "identity_verification",
    "escalation_request",
    "other",
]

ALLOWED_ROUTES = [
    "auto_resolve",
    "payments",
    "compliance",
    "risk",
    "retention",
    "legal",
]

ALLOWED_PRIORITIES = ["P1", "P2", "P3", "P4"]

ALLOWED_COACHING_TECHNIQUES = [
    "expectation_setting",
    "ownership_language",
    "proactive_disclosure",
    "empathy_with_resolution",
    "clear_next_step",
    "avoid_repetition",
    "complaint_handling",
    "policy_explanation",
]

# Maps native-language speaker labels to normalised roles.
SPEAKER_NORMALISATION = {
    # T1 — English
    "User": "user",
    "Agent": "agent",
    # T2 — Spanish
    "Usuario": "user",
    "Agente": "agent",
    # T3 — Bahasa Indonesia
    "Pengguna": "user",
    "Agen": "agent",
    # T4 — Mandarin
    "用户": "user",
    "客服": "agent",
    # T5 — Arabic
    "المستخدم": "user",
    "الموظف": "agent",
}

# Maps trigger phrase → (bcp47_language_code, recommended_escalation).
# Latin-script entries are matched case-insensitively; CJK/Arabic matched as-is.
COMPLIANCE_KEYWORDS = {
    # English
    "complaint": ("en", "compliance"),
    "fraud": ("en", "legal"),
    "scam": ("en", "legal"),
    "frozen account": ("en", "risk"),
    "withdraw": ("en", "compliance"),
    "investigate": ("en", "compliance"),
    "legal": ("en", "legal"),
    # Spanish
    "queja": ("es", "compliance"),
    "fraude": ("es", "legal"),
    "retirar": ("es", "compliance"),
    # Bahasa Indonesia
    "penipuan": ("id", "legal"),
    "rugi": ("id", "risk"),
    "paksa": ("id", "risk"),
    # Mandarin (matched as exact substring)
    "投诉": ("zh", "compliance"),
    "冻结": ("zh", "risk"),
    "欺诈": ("zh", "legal"),
    # Arabic (matched as exact substring)
    "احتيال": ("ar", "legal"),
    "شكوى": ("ar", "compliance"),
    "سحب": ("ar", "compliance"),
}

# Languages that use Latin script — matched case-insensitively.
LATIN_SCRIPT_LANGS = {"en", "es", "id"}

# Speaker label sets used by normalize_speaker() for case-insensitive lookup.
USER_LABELS = {"user", "usuario", "pengguna", "用户", "المستخدم"}
AGENT_LABELS = {"agent", "agente", "agen", "客服", "الموظف"}


def normalize_speaker(speaker: str) -> str:
    s = speaker.strip().lower()
    if s in USER_LABELS:
        return "user"
    if s in AGENT_LABELS:
        return "agent"
    return "unknown"


def is_user(speaker: str) -> bool:
    return normalize_speaker(speaker) == "user"


def is_agent(speaker: str) -> bool:
    return normalize_speaker(speaker) == "agent"
