from types import SimpleNamespace
from typing import Any

import pytest


class MockBackend:
    def __init__(self):
        self.messages = []

    def _add_json_in_prompt(self, new_messages):
        self.messages.append("JSON_ADDED")


def test_json_added_once():
    backend = MockBackend()
    try_n = 3
    json_added = False
    new_messages = ["msg1"]

    for _ in range(try_n):
        if not json_added:
            backend._add_json_in_prompt(new_messages)
            json_added = True

    assert backend.messages.count("JSON_ADDED") == 1


@pytest.mark.offline
def test_litellm_settings_logging_excludes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    module = pytest.importorskip("rdagent.oai.backend.litellm")
    infos = []
    objects = []
    marker = "synthetic-private-value"
    fields = {"chat_model": "fixture-model", "embedding_model": "fixture-embedding",
              "reasoning_effort": None, "chat_stream": False, "enable_response_schema": True,
              "openai_api_key": marker, "custom_endpoint_key": marker, "openai_api_base": marker}

    class Settings:
        def model_dump(self, *, include: set[str] | None = None) -> dict[str, Any]:
            return {key: value for key, value in fields.items() if include is None or key in include}

        def __str__(self) -> str:
            return marker

    monkeypatch.setattr(module, "LITELLM_SETTINGS", Settings())
    monkeypatch.setattr(module, "logger", SimpleNamespace(
        info=infos.append, log_object=lambda value, **kwargs: objects.append((value, kwargs))))
    monkeypatch.setattr(module.APIBackend, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.LiteLLMAPIBackend, "_has_logged_settings", False)
    module.LiteLLMAPIBackend()
    module.LiteLLMAPIBackend()
    assert len(infos) == len(objects) == 1
    assert marker not in str(infos)
    assert marker not in str(objects)
    assert objects[0][0]["chat_model"] == "fixture-model"
    assert "openai_api_key" not in objects[0][0]
