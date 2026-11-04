import os
from unittest.mock import MagicMock
import pytest
import requests  # type: ignore[import-untyped]

from app.agents.llm_client import (
    NvidiaClientConfig,
    NvidiaAuthenticationError,
    NvidiaTimeoutError,
    NvidiaConnectionError,
    NvidiaInferenceError,
    get_llm,
    invoke_llm,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)


def test_config_from_env_defaults(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key-1234567890")
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_TIMEOUT", raising=False)
    monkeypatch.delenv("NVIDIA_TEMPERATURE", raising=False)
    monkeypatch.delenv("NVIDIA_MAX_TOKENS", raising=False)

    config = NvidiaClientConfig.from_env(load_env=False)

    assert config.api_key == "nvapi-test-key-1234567890"
    assert config.model == DEFAULT_NVIDIA_MODEL
    assert config.timeout == DEFAULT_TIMEOUT
    assert config.temperature == DEFAULT_TEMPERATURE
    assert config.max_completion_tokens == DEFAULT_MAX_TOKENS


def test_config_from_env_custom(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-custom-key")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.2-3b-instruct")
    monkeypatch.setenv("NVIDIA_TIMEOUT", "45.5")
    monkeypatch.setenv("NVIDIA_TEMPERATURE", "0.2")
    monkeypatch.setenv("NVIDIA_MAX_TOKENS", "512")

    config = NvidiaClientConfig.from_env(load_env=False)

    assert config.api_key == "nvapi-test-custom-key"
    assert config.model == "meta/llama-3.2-3b-instruct"
    assert config.timeout == 45.5
    assert config.temperature == 0.2
    assert config.max_completion_tokens == 512


def test_config_masks_api_key():
    config = NvidiaClientConfig(
        api_key="nvapi-super-secret-key-12345",
        model="test-model"
    )
    repr_str = repr(config)
    assert "super-secret" not in repr_str
    assert "nvapi-" in repr_str
    assert "2345" in repr_str


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(NvidiaAuthenticationError, match="NVIDIA_API_KEY is not set"):
        NvidiaClientConfig.from_env(load_env=False)


def test_get_llm_builds_chat_nvidia():
    config = NvidiaClientConfig(
        api_key="nvapi-mock-key",
        model="meta/llama-3.1-8b-instruct",
        temperature=0.0,
        max_completion_tokens=100,
        timeout=10.0
    )

    llm = get_llm(config)
    assert llm.model == "meta/llama-3.1-8b-instruct"
    assert llm.temperature == 0.0
    assert getattr(llm, "max_tokens", None) == 100 or getattr(llm, "max_completion_tokens", None) == 100


def test_invoke_llm_wraps_timeout():
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.invoke.side_effect = requests.exceptions.Timeout("Request timed out")

    with pytest.raises(NvidiaTimeoutError, match="timed out"):
        invoke_llm(mock_llm, "test input")


def test_invoke_llm_wraps_connection_error():
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.invoke.side_effect = requests.exceptions.ConnectionError("Failed to connect")

    with pytest.raises(NvidiaConnectionError, match="Failed to connect"):
        invoke_llm(mock_llm, "test input")


def test_invoke_llm_wraps_authorization_error():
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.api_key = "nvapi-secret-key-value"
    mock_llm.invoke.side_effect = Exception("[403] Forbidden: Authorization failed")

    with pytest.raises(NvidiaAuthenticationError, match="HTTP 403"):
        invoke_llm(mock_llm, "test input")


def test_no_silent_model_fallback():
    """Ensure that failure on a model directly propagates error rather than switching models."""
    mock_llm = MagicMock()
    mock_llm.model = "primary-model"
    mock_llm.invoke.side_effect = Exception("Model unavailable")

    with pytest.raises(NvidiaInferenceError) as exc_info:
        invoke_llm(mock_llm, "test input")

    assert mock_llm.model == "primary-model"
    assert "Model unavailable" in str(exc_info.value)
