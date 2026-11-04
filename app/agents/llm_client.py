import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import requests  # type: ignore[import-untyped]
import urllib3.exceptions


load_dotenv()

DEFAULT_NVIDIA_MODEL = "meta/muse-glimmer-30b"
DEFAULT_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 300


class NvidiaClientError(Exception):
    """Base exception for NVIDIA NIM client errors."""
    pass


class NvidiaAuthenticationError(NvidiaClientError):
    """Raised when the NVIDIA API key is missing, invalid, or unauthorized."""
    pass


class NvidiaInferenceError(NvidiaClientError):
    """Raised when an error occurs during model inference."""
    pass


class NvidiaTimeoutError(NvidiaInferenceError):
    """Raised when an inference request times out."""
    pass


class NvidiaConnectionError(NvidiaInferenceError):
    """Raised when a network connection to NVIDIA NIM fails."""
    pass


@dataclass
class NvidiaClientConfig:
    """
    Configuration parameters for the NVIDIA NIM client.
    Guarantees that credentials are not exposed in representations or logs.
    """
    api_key: str
    model: str = DEFAULT_NVIDIA_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_completion_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    base_url: str | None = None

    def __repr__(self) -> str:
        masked_key = (
            f"{self.api_key[:6]}...{self.api_key[-4:]}"
            if len(self.api_key) > 10
            else "***"
        )
        return (
            f"NvidiaClientConfig("
            f"model={self.model!r}, "
            f"temperature={self.temperature}, "
            f"max_completion_tokens={self.max_completion_tokens}, "
            f"timeout={self.timeout}, "
            f"base_url={self.base_url!r}, "
            f"api_key={masked_key!r})"
        )

    @classmethod
    def from_env(cls, load_env: bool = True, **overrides: Any) -> "NvidiaClientConfig":
        """
        Build client configuration from environment variables with parameter overrides.
        """
        if load_env:
            load_dotenv()

        api_key = str(overrides.get("api_key") or os.getenv("NVIDIA_API_KEY") or "")

        if not api_key:
            raise NvidiaAuthenticationError(
                "NVIDIA_API_KEY is not set in environment or .env file."
            )

        model_env = overrides.get("model") or os.getenv(
            "NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL
        )
        model = str(model_env) if model_env else DEFAULT_NVIDIA_MODEL

        try:
            temp_env = overrides.get("temperature") or os.getenv("NVIDIA_TEMPERATURE")
            temperature = (
                float(temp_env) if temp_env is not None else DEFAULT_TEMPERATURE
            )
        except ValueError:
            temperature = DEFAULT_TEMPERATURE

        try:
            tokens_env = overrides.get("max_completion_tokens") or os.getenv("NVIDIA_MAX_TOKENS")
            max_completion_tokens = (
                int(tokens_env) if tokens_env is not None else DEFAULT_MAX_TOKENS
            )
        except ValueError:
            max_completion_tokens = DEFAULT_MAX_TOKENS

        try:
            timeout_env = overrides.get("timeout") or os.getenv("NVIDIA_TIMEOUT")
            timeout = (
                float(timeout_env) if timeout_env is not None else DEFAULT_TIMEOUT
            )
        except ValueError:
            timeout = DEFAULT_TIMEOUT

        base_url_env = overrides.get("base_url") or os.getenv("NVIDIA_BASE_URL")
        base_url = str(base_url_env) if base_url_env else None

        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            base_url=base_url
        )


def get_llm(
    config: NvidiaClientConfig | None = None,
    **overrides: Any
) -> ChatNVIDIA:
    """
    Construct a ChatNVIDIA instance from configuration or environment variables.

    Deterministic settings (temperature=0.0) are applied by default.
    Does not implement silent model fallback.
    """
    if config is None:
        config = NvidiaClientConfig.from_env(**overrides)

    init_kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": config.api_key,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "timeout": config.timeout,
    }

    if config.base_url:
        init_kwargs["base_url"] = config.base_url

    return ChatNVIDIA(**init_kwargs)


def invoke_llm(llm: ChatNVIDIA, messages: Any) -> Any:
    """
    Safely invoke ChatNVIDIA, handling connection errors, timeouts, and API errors.

    Does not perform silent model fallback.
    Redacts credentials from error messages.
    """
    try:
        response = llm.invoke(messages)
        if hasattr(response, "content") and not response.content:
            extra = getattr(response, "additional_kwargs", {})
            reasoning = extra.get("reasoning_content") if isinstance(extra, dict) else None
            if reasoning:
                response.content = str(reasoning).strip()
        return response
    except NvidiaClientError:
        raise
    except (requests.exceptions.Timeout, urllib3.exceptions.TimeoutError) as err:
        raise NvidiaTimeoutError(
            f"NVIDIA NIM request timed out for model '{llm.model}' after configured timeout: {err}"
        ) from err
    except (
        requests.exceptions.ConnectionError,
        urllib3.exceptions.MaxRetryError,
        urllib3.exceptions.NameResolutionError
    ) as err:
        raise NvidiaConnectionError(
            f"Failed to connect to NVIDIA NIM endpoint for model '{llm.model}'. "
            "Please check network connectivity, proxy settings, or endpoint URL."
        ) from err
    except Exception as err:
        error_name = type(err).__name__
        raw_key = getattr(llm, "api_key", None)
        clean_msg = str(err)
        if raw_key and isinstance(raw_key, str):
            clean_msg = clean_msg.replace(raw_key, "[REDACTED]")

        if "403" in clean_msg or "Authorization failed" in clean_msg:
            raise NvidiaAuthenticationError(
                "NVIDIA NIM authorization failed (HTTP 403). "
                "The NVIDIA API key in .env is either invalid, lacks access to this endpoint, or its account credits have expired."
            ) from err

        if "404" in clean_msg or "not found" in clean_msg.lower():
            raise NvidiaInferenceError(
                f"Model '{llm.model}' was not found on the NVIDIA NIM endpoint (HTTP 404)."
            ) from err

        if "410" in clean_msg or "end of life" in clean_msg.lower():
            raise NvidiaInferenceError(
                f"Model '{llm.model}' has reached end-of-life on NVIDIA NIM (HTTP 410)."
            ) from err

        raise NvidiaInferenceError(
            f"NVIDIA NIM inference failed ({error_name}): {clean_msg}"
        ) from err


if __name__ == "__main__":
    try:
        llm = get_llm()
        print(f"Connecting to model '{llm.model}'...")
        response = invoke_llm(llm, "Explain prompt injection in two sentences.")
        print("\nNVIDIA MODEL RESPONSE:")
        print(response.content)
    except NvidiaClientError as exc:
        print(f"\n[NVIDIA CLIENT DIAGNOSTIC] {exc}")
