import re


REQUIRED_HEADINGS = (
    "## Executive Summary",
    "## Key Developments",
    "## Risks & Watchpoints",
    "## Bull / Base / Bear",
    "## Open Questions",
    "## Source Caveats",
)


def validate_agent_memo(memo_text: str) -> tuple[bool, str]:
    text = (memo_text or "").strip()
    if not text:
        return False, "No narrative was returned by the agent workflow."

    if len(text) > 12000:
        return False, "The returned narrative exceeded the display limit."

    lower_text = text.lower()
    if "<script" in lower_text or "<iframe" in lower_text:
        return False, "The returned narrative contained unsupported HTML."

    missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
    if missing:
        return False, f"The narrative is missing required sections: {', '.join(missing)}."

    return True, text


def normalize_agent_memo(memo_text: str) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", (memo_text or "")).strip()
    return cleaned
