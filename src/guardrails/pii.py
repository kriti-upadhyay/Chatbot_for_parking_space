"""
PII detection and anonymisation using Microsoft Presidio.

Detects: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IBAN_CODE,
         UK_VEHICLE_REG (custom), and other Presidio-supported entities.

Falls back gracefully to regex-only mode if the spaCy model is unavailable,
so the rest of the application keeps working with reduced PII coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import get_logger

logger = get_logger(__name__)

# Entities we actively look for; kept as a tuple for easy extension.
_ENTITIES_OF_INTEREST = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "UK_VEHICLE_REG",
)

# Minimum confidence threshold for a detection to count.
_MIN_SCORE = 0.5


# ---------------------------------------------------------------------------
# Custom recogniser — UK vehicle registration
# ---------------------------------------------------------------------------

def _build_vehicle_reg_recognizer():
    """Build a Presidio PatternRecognizer for UK vehicle registrations."""
    from presidio_analyzer import Pattern, PatternRecognizer

    patterns = [
        Pattern(
            "UK_VEHICLE_REG_POST2001",
            r"\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b",
            0.8,
        ),
        Pattern(
            "UK_VEHICLE_REG_PREFIX",
            r"\b[A-Z][0-9]{1,3}\s?[A-Z]{3}\b",
            0.6,
        ),
        Pattern(
            "UK_VEHICLE_REG_SUFFIX",
            r"\b[A-Z]{3}\s?[0-9]{1,3}[A-Z]\b",
            0.6,
        ),
    ]
    return PatternRecognizer(
        supported_entity="UK_VEHICLE_REG",
        patterns=patterns,
        name="UKVehicleRegRecognizer",
    )


# ---------------------------------------------------------------------------
# Analyzer factory
# ---------------------------------------------------------------------------

def _build_analyzer():
    """
    Attempt to build a Presidio AnalyzerEngine with the spaCy NLP backend.

    Tries en_core_web_sm; falls back to a regex-only configuration if the
    model is not installed.  Returns (analyzer, is_full_nlp).
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    vehicle_reg_recognizer = _build_vehicle_reg_recognizer()

    # ── Try spaCy-backed engine ───────────────────────────────────────────
    for model_name in ("en_core_web_sm", "en_core_web_lg"):
        try:
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            registry.add_recognizer(vehicle_reg_recognizer)
            analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry,
            )
            logger.info("Presidio initialized with spaCy model: %s", model_name)
            return analyzer, True
        except OSError:
            logger.debug("spaCy model '%s' not found, trying next.", model_name)
        except Exception as exc:
            logger.debug("Failed to init Presidio with %s: %s", model_name, exc)

    # ── Fallback: regex-only (no spaCy NER) ──────────────────────────────
    logger.warning(
        "No spaCy model available. Presidio running in regex-only mode. "
        "PERSON entity detection will be limited. "
        "Run: python -m spacy download en_core_web_sm"
    )
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(vehicle_reg_recognizer)
    analyzer = AnalyzerEngine(registry=registry)
    return analyzer, False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PiiCheckResult:
    has_pii: bool
    entity_types: list[str]
    anonymized_text: str
    original_text: str
    presidio_available: bool


# ---------------------------------------------------------------------------
# PiiDetector
# ---------------------------------------------------------------------------

class PiiDetector:
    """
    Detect and anonymise PII in text.

    Initialise once and reuse — spaCy model loading is expensive.
    """

    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None
        self._available = False
        self._nlp_full = False
        self._init()

    def _init(self) -> None:
        try:
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer, self._nlp_full = _build_analyzer()
            self._anonymizer = AnonymizerEngine()
            self._available = True
            logger.info(
                "PiiDetector ready (full_nlp=%s)", self._nlp_full
            )
        except ImportError:
            logger.warning(
                "presidio-analyzer or presidio-anonymizer not installed. "
                "PII detection is disabled."
            )
        except Exception:
            logger.exception("PiiDetector initialisation failed.")

    @property
    def available(self) -> bool:
        return self._available

    def check(self, text: str, language: str = "en") -> PiiCheckResult:
        """
        Analyse *text* and return a PiiCheckResult.

        If Presidio is unavailable, returns a safe default (no PII detected,
        original text unchanged) so the calling code can proceed.
        """
        if not text or not self._available:
            return PiiCheckResult(
                has_pii=False,
                entity_types=[],
                anonymized_text=text,
                original_text=text,
                presidio_available=self._available,
            )

        try:
            results = self._analyzer.analyze(
                text=text,
                language=language,
                entities=list(_ENTITIES_OF_INTEREST),
                score_threshold=_MIN_SCORE,
            )

            if not results:
                return PiiCheckResult(
                    has_pii=False,
                    entity_types=[],
                    anonymized_text=text,
                    original_text=text,
                    presidio_available=True,
                )

            entity_types = list({r.entity_type for r in results})
            anonymized = self._anonymizer.anonymize(
                text=text, analyzer_results=results
            )

            return PiiCheckResult(
                has_pii=True,
                entity_types=entity_types,
                anonymized_text=anonymized.text,
                original_text=text,
                presidio_available=True,
            )
        except Exception:
            logger.exception("PiiDetector.check() failed for input: %.50s", text)
            return PiiCheckResult(
                has_pii=False,
                entity_types=[],
                anonymized_text=text,
                original_text=text,
                presidio_available=True,
            )

    def scrub_output(self, text: str) -> str:
        """
        Remove unexpected PII from LLM-generated text before showing the user.

        Reservation data (names, reg numbers) may appear in the LLM context
        window through conversation history; this prevents accidental leakage
        in general responses.
        """
        if not text or not self._available:
            return text
        result = self.check(text)
        if result.has_pii:
            logger.warning(
                "PII found in LLM output — scrubbing entities: %s",
                result.entity_types,
            )
            return result.anonymized_text
        return text


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_pii_detector: PiiDetector | None = None


def get_pii_detector() -> PiiDetector:
    """Return the module-level PiiDetector singleton (lazy init)."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PiiDetector()
    return _pii_detector
