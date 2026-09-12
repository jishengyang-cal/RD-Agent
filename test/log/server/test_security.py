from collections import defaultdict
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import rdagent.log.server.app as server
import rdagent.log.ui.storage as web_storage
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log.server import debug_app
from rdagent.log.server.security import (
    parse_competition,
    resolve_within,
    validate_scenario,
    validate_upload_filename,
)
from rdagent.log.storage import FileStorage


@pytest.mark.offline
@pytest.mark.parametrize("credential", [None, "header", "cookie"])
def test_debug_rejects_unauthenticated_replay_without_side_effects(monkeypatch, credential):
    monkeypatch.setitem(debug_app.app.config, "AUTH_TOKEN", "fixture-token")
    messages = defaultdict(list, {"fixture": [{"tag": "fixture", "content": "private"}]})
    pointers = defaultdict(int, {"fixture": 0})
    monkeypatch.setattr(debug_app, "msgs_for_frontend", messages)
    monkeypatch.setattr(debug_app, "pointers", pointers)

    def forbidden(*args, **kwargs):
        pytest.fail("Unauthenticated request attempted trace loading or thread creation")

    monkeypatch.setattr(debug_app, "resolve_within", forbidden)
    monkeypatch.setattr(debug_app.threading, "Thread", forbidden)
    monkeypatch.setattr(FileStorage, "iter_msg", forbidden)
    client = debug_app.app.test_client()
    headers = {"Authorization": "Bearer wrong-token"} if credential == "header" else {}
    if credential == "cookie":
        client.set_cookie("rdagent_auth", "wrong-token")
    responses = [
        client.post("/upload", data={"scenario": "Finance Data Building"}, headers=headers),
        client.post("/trace", json={"id": "fixture", "all": True}, headers=headers),
        client.post("/receive", json={"id": "fixture", "msg": {"tag": "injected"}}, headers=headers),
        client.post("/control", json={"id": "fixture", "action": "stop"}, headers=headers),
        client.get("/test", headers=headers),
    ]
    for response in responses:
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.get_json() == {"error": "Authentication required"}
    assert dict(messages) == {"fixture": [{"tag": "fixture", "content": "private"}]}
    assert dict(pointers) == {"fixture": 0}


@pytest.mark.offline
@pytest.mark.parametrize("credential", ["header", "cookie", "unconfigured"])
def test_debug_reads_authorized_synthetic_fixture(monkeypatch, tmp_path, credential):
    monkeypatch.setitem(debug_app.app.config, "AUTH_TOKEN", "" if credential == "unconfigured" else "fixture-token")
    monkeypatch.setattr(debug_app.UI_SETTING, "trace_folder", str(tmp_path / "traces"))
    monkeypatch.setattr(debug_app, "msgs_for_frontend", defaultdict(list))
    monkeypatch.setattr(debug_app, "pointers", defaultdict(int))
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "synthetic-fixture-key-for-security-tests")
    storage = FileStorage(tmp_path / "traces" / "Finance Data Building")
    storage.log(SimpleNamespace(experiment_setting="synthetic authorized trace"), tag="scenario.fixture")
    starts = []

    def synchronous_thread(*, target, args, daemon):
        def start():
            starts.append(args[0])
            target(*args)

        return SimpleNamespace(start=start)

    monkeypatch.setattr(debug_app.threading, "Thread", synchronous_thread)
    monkeypatch.setattr(debug_app.time, "sleep", lambda _: None)
    client = debug_app.app.test_client()
    headers = {"Authorization": "Bearer fixture-token"} if credential == "header" else {}
    if credential == "cookie":
        client.set_cookie("rdagent_auth", "fixture-token")
    response = client.post("/upload", data={"scenario": "Finance Data Building"}, headers=headers)
    assert response.status_code == HTTPStatus.OK
    assert starts == [storage.path.resolve()]
    response = client.post("/trace", json={"id": response.get_json()["id"], "all": True}, headers=headers)
    assert response.status_code == HTTPStatus.OK
    messages = response.get_json()
    assert len(messages) == 2
    assert messages[0]["tag"] == "feedback.config"
    assert messages[0]["content"] == {"config": "synthetic authorized trace"}
    assert messages[1]["tag"] == "END"


@pytest.mark.offline
@pytest.mark.parametrize("application", [server.app, debug_app.app])
def test_servers_share_cookie_and_public_route_contract(monkeypatch, application, tmp_path):
    monkeypatch.setitem(application.config, "AUTH_TOKEN", "fixture-token")
    (tmp_path / "index.html").write_text("synthetic public shell")
    monkeypatch.setattr(application, "static_folder", str(tmp_path))
    client = application.test_client()
    assert client.get("/").status_code == HTTPStatus.OK
    assert client.options("/trace").status_code == HTTPStatus.OK
    assert client.get("/test").status_code == HTTPStatus.UNAUTHORIZED
    client.set_cookie("rdagent_auth", "fixture-token")
    assert client.get("/test").status_code == HTTPStatus.OK
    assert client.get("/test", headers={"Authorization": "Bearer wrong-token"}).status_code == HTTPStatus.UNAUTHORIZED
    client.delete_cookie("rdagent_auth")
    monkeypatch.setitem(application.config, "AUTH_TOKEN", "")
    assert client.get("/test").status_code == HTTPStatus.OK


class _Response:
    status_code = HTTPStatus.OK
    text = "ok"


@pytest.mark.offline
def test_debug_trace_uses_configured_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setitem(debug_app.app.config, "AUTH_TOKEN", "")
    monkeypatch.setattr(debug_app.UI_SETTING, "trace_folder", str(tmp_path / "traces"))
    monkeypatch.setattr(
        debug_app.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: calls.append(kwargs)),
    )
    response = debug_app.app.test_client().post("/upload", data={"scenario": "Finance Data Building"})
    assert response.status_code == HTTPStatus.OK
    assert calls[0]["args"][0] == (tmp_path / "traces" / "Finance Data Building").resolve()


@pytest.mark.offline
def test_validate_scenario_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        validate_scenario("../Data Science")


@pytest.mark.offline
def test_parse_competition_accepts_only_mle_bench_slug() -> None:
    assert parse_competition("MLE-Bench:aerial-cactus-identification") == "aerial-cactus-identification"
    for value in (None, "aerial-cactus", "MLE-Bench:../../tmp", "MLE-Bench:a;id"):
        with pytest.raises(ValueError, match=r"Competition|Invalid"):
            parse_competition(value)


@pytest.mark.offline
def test_resolve_within_rejects_escape(tmp_path: Path) -> None:
    assert resolve_within(tmp_path, "scenario", "trace").is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        resolve_within(tmp_path, "..", "outside")


@pytest.mark.offline
def test_upload_filename_rejects_executable_formats() -> None:
    assert validate_upload_filename("report.pdf") == "report.pdf"
    for filename in ("payload.pkl", "payload.PICKLE", "script.py", ""):
        with pytest.raises(ValueError, match=r"upload|file type"):
            validate_upload_filename(filename)


@pytest.mark.offline
def test_log_server_requires_authentication_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "secret-token")
    client = server.app.test_client()

    assert client.get("/traces").status_code == HTTPStatus.UNAUTHORIZED
    response = client.get("/traces", headers={"Authorization": "Bearer secret-token"})
    assert response.status_code == HTTPStatus.OK


@pytest.mark.offline
@pytest.mark.parametrize(
    ("token", "expected_authorization"),
    [("secret-token", "Bearer secret-token"), ("", None)],
)
def test_web_storage_authenticates_internal_receive_requests(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    expected_authorization: str | None,
) -> None:
    request_headers: dict[str, str] = {}

    def fake_post(url: str, *, json: object, headers: dict[str, str], timeout: int) -> _Response:
        assert url == "http://localhost:19899/receive"
        assert json == {"id": "trace", "msg": {"tag": "test"}}
        assert timeout == 1
        request_headers.update(headers)
        return _Response()

    monkeypatch.setattr(web_storage.UI_SETTING, "server_auth_token", token)
    monkeypatch.setattr(web_storage.requests, "post", fake_post)
    storage = web_storage.WebStorage(port=19899, path="trace")
    monkeypatch.setattr(
        storage,
        "_obj_to_json",
        lambda **_kwargs: {"id": "trace", "msg": {"tag": "test"}},
    )

    assert storage.log("message", "test") == "200 ok"
    assert request_headers.get("Authorization") == expected_authorization


@pytest.mark.offline
def test_upload_rejects_unknown_scenario_before_writing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "")
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    client = server.app.test_client()

    response = client.post("/upload", data={"scenario": "../Data Science"})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.offline
def test_upload_rejects_pickle_before_starting_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "")
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    monkeypatch.setattr(server, "log_folder_path", tmp_path / "traces")
    client = server.app.test_client()

    response = client.post(
        "/upload",
        data={"scenario": "Finance Data Building", "files": (BytesIO(b"payload"), "payload.pkl")},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not (tmp_path / "uploads").exists()
