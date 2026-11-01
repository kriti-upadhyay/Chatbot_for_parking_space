"""
Prompt injection detection.

Uses a scored pattern-matching approach: multiple independent signals each
contribute a suspicion score; if the total exceeds a threshold the input
is flagged as a likely injection attempt.

This is more robust than a single regex because a sophisticated attacker
would avoid any one pattern but would still trigger several at lower
confidence.
"""

import re
from dataclasses import dataclass, field

from src.config import get_logger

logger = get_logger(__name__)

_INJECTION_THRESHOLD = 0.5  # combined score required to flag as injection


@dataclass
class InjectionCheckResult:
    is_injection: bool
    score: float           # 0.0 – 1.0
    triggered_patterns: list[str] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Scored pattern registry
# Each entry: (label, compiled_regex, score_contribution)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    (
        "role_change",
        re.compile(
            r"(act\s+as|you\s+are\s+now|pretend\s+to\s+be|take\s+on\s+the\s+role|"
            r"roleplay|new\s+persona|switch\s+to)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "instruction_override",
        re.compile(
            r"(ignore|disregard|forget|override|bypass)\s+"
            r"(previous|all|above|prior|your|these)\s*(instructions?|rules?|guidelines?|prompts?)?",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        "system_manipulation",
        re.compile(
            r"(system\s+(prompt|message|instructions?|config)|jailbreak|DAN\b|"
            r"developer\s+mode|god\s+mode)",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        "privilege_escalation",
        re.compile(
            r"(admin(istrator)?\s+(mode|access|privilege|override)|"
            r"superuser|root\s+access|unrestricted\s+mode)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_extraction",
        re.compile(
            r"(reveal|show|print|display|output|dump|leak)\s+"
            r"(system|prompt|instructions?|config|training\s+data|context)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "instruction_injection",
        re.compile(
            r"(new\s+instructions?|updated\s+instructions?|"
            r"your\s+new\s+(task|goal|objective|directive)|"
            r"from\s+now\s+on\s+you\s+(will|must|should))",
            re.IGNORECASE,
        ),
        0.4,
    ),
    (
        "separator_trick",
        # Attempts to end the system prompt and inject a new one
        re.compile(r"(---+|===+|\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<SYS>|</SYS>)"),
        0.3,
    ),
]


class InjectionDetector:
    """Detect prompt injection attempts using a scored multi-pattern approach."""

    def check(self, text: str) -> InjectionCheckResult:
        """
        Analyse *text* for injection patterns.

        Returns:
            InjectionCheckResult with is_injection=True if total score
            exceeds _INJECTION_THRESHOLD.
        """
        if not text or not text.strip():
            return InjectionCheckResult(is_injection=False, score=0.0)

        total_score = 0.0
        triggered: list[str] = []

        for label, pattern, score in _PATTERNS:
            if pattern.search(text):
                total_score = min(1.0, total_score + score)
                triggered.append(label)
                logger.debug("Injection pattern matched: %s (score=%.2f)", label, score)

        is_injection = total_score >= _INJECTION_THRESHOLD

        reason = ""
        if is_injection:
            reason = f"Triggered: {', '.join(triggered)} (score={total_score:.2f})"
            logger.warning("Injection attempt detected | %s | %.80s", reason, text)

        return InjectionCheckResult(
            is_injection=is_injection,
            score=total_score,
            triggered_patterns=triggered,
            reason=reason,
        )


# Module-level singleton
_detector: InjectionDetector | None = None


def get_injection_detector() -> InjectionDetector:
    global _detector
    if _detector is None:
        _detector = InjectionDetector()
    return _detector
