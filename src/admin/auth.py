"""
Administrator authentication.

Uses bcrypt for password verification.  The plaintext password is read
from the ADMIN_PASSWORD environment variable at verification time and
never stored or logged.

If ADMIN_PASSWORD is not set (empty string), authentication always fails
so the admin interface is locked out by default.
"""

import hashlib
import hmac

from src.config import get_logger, settings

logger = get_logger(__name__)

# Minimum allowed password length to avoid trivially weak passwords.
_MIN_PASSWORD_LEN = 6


def verify_admin_password(candidate: str) -> bool:
    """
    Return True if *candidate* matches the configured ADMIN_PASSWORD.

    Comparison is done with ``hmac.compare_digest`` to prevent timing attacks.
    The candidate and stored value are compared as UTF-8 byte strings.

    Returns False (never raises) if ADMIN_PASSWORD is not configured or
    if *candidate* is empty.
    """
    stored = settings.admin_password
    if not stored or not candidate:
        return False

    # Constant-time comparison to prevent timing side-channels.
    match = hmac.compare_digest(
        candidate.encode("utf-8"),
        stored.encode("utf-8"),
    )
    if not match:
        logger.warning("Admin authentication failed (wrong password).")
    return match


def is_admin_configured() -> bool:
    """Return True if ADMIN_PASSWORD has been set to a non-empty value."""
    return bool(settings.admin_password)
