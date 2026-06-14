"""Gateway ``/model`` (no args) reports the model the cached agent is
*actually* talking to when a fallback is active, not the config primary
that fell over.  Lost in the v0.15.0 refactor (#45970).
"""

from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._agent_cache = {}
    import threading
    runner._agent_cache_lock = threading.RLock()
    return runner


def _make_event(text, chat_id="12345"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
    )


def _isolated_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"model": {"default": "primary-model", "provider": "primary-prov"}, "providers": {}}
        ),
        encoding="utf-8",
    )
    import gateway.run as gateway_run
    import hermes_cli.config
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(hermes_cli.config, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


def _make_agent(*, model, provider, fallback_activated):
    return SimpleNamespace(
        _fallback_activated=fallback_activated, model=model, provider=provider
    )


class TestModelCommandFallbackActive:
    @pytest.mark.asyncio
    async def test_no_fallback_shows_configured_primary(self, tmp_path, monkeypatch):
        _isolated_home(tmp_path, monkeypatch)
        runner = _make_runner()
        session_key = runner._session_key_for_source(
            SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
        )
        runner._agent_cache[session_key] = (
            _make_agent(model="primary-model", provider="primary-prov", fallback_activated=False),
            None,
        )

        result = await runner._handle_model_command(_make_event("/model"))

        assert "primary-model" in result
        assert "fallback" not in result.lower()

    @pytest.mark.asyncio
    async def test_fallback_active_shows_active_model_and_warning(
        self, tmp_path, monkeypatch
    ):
        _isolated_home(tmp_path, monkeypatch)
        runner = _make_runner()
        session_key = runner._session_key_for_source(
            SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
        )
        runner._agent_cache[session_key] = (
            _make_agent(
                model="fallback-model",
                provider="fallback-prov",
                fallback_activated=True,
            ),
            None,
        )

        result = await runner._handle_model_command(_make_event("/model"))

        assert "fallback-model" in result
        assert "fallback" in result.lower()

    @pytest.mark.asyncio
    async def test_no_cached_agent_falls_back_to_config(self, tmp_path, monkeypatch):
        _isolated_home(tmp_path, monkeypatch)
        runner = _make_runner()

        result = await runner._handle_model_command(_make_event("/model"))

        assert "primary-model" in result
        assert "fallback" not in result.lower()

    @pytest.mark.asyncio
    async def test_cached_agent_without_fallback_attr_works(self, tmp_path, monkeypatch):
        _isolated_home(tmp_path, monkeypatch)
        runner = _make_runner()
        session_key = runner._session_key_for_source(
            SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
        )
        # Older session: cached agent predates the _fallback_activated attr.
        runner._agent_cache[session_key] = (
            SimpleNamespace(model="primary-model", provider="primary-prov"),
            None,
        )

        result = await runner._handle_model_command(_make_event("/model"))

        assert "primary-model" in result
        assert "fallback" not in result.lower()
