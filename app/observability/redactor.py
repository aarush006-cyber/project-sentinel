import re
from typing import Any


class SecretRedactor:
    """
    Sanitizes structured logs and text by masking API keys, credentials,
    and sensitive parameters to prevent accidental information leakage.
    """

    SENSITIVE_KEY_PATTERNS = {
        "api_key",
        "apikey",
        "key",
        "token",
        "auth",
        "authorization",
        "password",
        "secret",
        "credentials",
        "bearer",
        "private_key",
    }

    VALUE_REGEX_REPLACEMENTS = [
        (re.compile(r"nvapi-[A-Za-z0-9_\-]{16,}", re.IGNORECASE), "nvapi-[REDACTED]"),
        (re.compile(r"sk-[A-Za-z0-9_\-]{20,}", re.IGNORECASE), "sk-[REDACTED]"),
        (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{10,}", re.IGNORECASE), "Bearer [REDACTED]"),
        (re.compile(r"(NVIDIA_API_KEY\s*=\s*)[^\s;]+", re.IGNORECASE), r"\1[REDACTED]"),
        (re.compile(r"(api_key\s*=\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED]\2"),
        (re.compile(r"(password\s*=\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED]\2"),
    ]

    @classmethod
    def redact(cls, data: Any) -> Any:
        """
        Recursively scrubs sensitive keys and patterns from dicts, lists, and strings.
        """
        if isinstance(data, dict):
            sanitized: dict[str, Any] = {}
            for k, v in data.items():
                k_lower = str(k).lower()
                if any(sens in k_lower for sens in cls.SENSITIVE_KEY_PATTERNS):
                    sanitized[k] = "[REDACTED]"
                else:
                    sanitized[k] = cls.redact(v)
            return sanitized

        elif isinstance(data, list):
            return [cls.redact(item) for item in data]

        elif isinstance(data, str):
            text = data
            for pattern, replacement in cls.VALUE_REGEX_REPLACEMENTS:
                text = pattern.sub(replacement, text)
            return text

        return data

