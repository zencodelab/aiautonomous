"""Tests for configuration loading and friendly error messages."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from taskengine.config import ConfigurationError, Settings, get_settings


def _make_validation_error(fields: list[str]) -> ValidationError:
    """Build a pydantic ValidationError for the given missing field names."""
    errors = [
        {
            "type": "missing",
            "loc": (field,),
            "msg": "Field required",
            "input": {},
            "url": "https://errors.pydantic.dev/2/v/missing",
        }
        for field in fields
    ]
    # pydantic v2 raises via PydanticUserError; use from_exception_data instead
    return ValidationError.from_exception_data(
        title="Settings",
        input_type="python",
        line_errors=[
            {
                "type": "missing",
                "loc": (field,),
                "msg": "Field required",
                "input": {},
                "url": "",
                "ctx": {},
            }
            for field in fields
        ],
    )


def test_get_settings_missing_keys_raises_configuration_error():
    """Missing required env vars produce ConfigurationError with helpful hints."""
    # Build a real ValidationError by passing an empty dict to Settings
    # (bypass env-file loading by directly creating the error via Pydantic)
    with patch("taskengine.config.Settings", side_effect=_make_missing_validation_error(["openai_api_key", "pinecone_api_key"])):
        with pytest.raises(ConfigurationError) as exc_info:
            get_settings()

    msg = str(exc_info.value)
    assert "OPENAI_API_KEY" in msg
    assert "PINECONE_API_KEY" in msg
    assert ".env" in msg


def test_get_settings_missing_openai_key_only():
    """Only the missing key is reported."""
    with patch("taskengine.config.Settings", side_effect=_make_missing_validation_error(["openai_api_key"])):
        with pytest.raises(ConfigurationError) as exc_info:
            get_settings()

    msg = str(exc_info.value)
    assert "OPENAI_API_KEY" in msg
    assert "PINECONE_API_KEY" not in msg


def test_get_settings_with_all_required_keys(monkeypatch):
    """When required vars are present, Settings loads without error."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")

    settings = get_settings()
    assert settings.openai_api_key == "sk-test"
    assert settings.pinecone_api_key == "pc-test"


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_missing_validation_error(fields: list[str]) -> ValidationError:
    """Return a ValidationError that looks like pydantic-settings missing fields."""
    from pydantic import TypeAdapter

    class _Dummy(Settings.__bases__[0]):  # inherit from BaseSettings
        model_config = Settings.model_config  # type: ignore[attr-defined]

    # Trigger a real ValidationError by calling the model with empty input
    try:
        Settings.model_validate({})
    except ValidationError as exc:
        # Filter to only the requested fields so tests are predictable
        filtered = [e for e in exc.errors() if e.get("loc") and e["loc"][0] in fields]
        if filtered:
            return _build_exc(filtered)
    # Fallback: build a synthetic error dict
    return _build_exc(
        [{"type": "missing", "loc": (f,), "msg": "Field required", "input": {}} for f in fields]
    )


def _build_exc(errors: list[dict]) -> ValidationError:
    """Re-raise a subset of errors as a new ValidationError."""
    from pydantic_core import InitErrorDetails, PydanticCustomError

    details = [
        InitErrorDetails(type="missing", loc=e["loc"], input=e.get("input", {}))
        for e in errors
    ]
    return ValidationError.from_exception_data(
        title="Settings",
        input_type="python",
        line_errors=details,
    )
