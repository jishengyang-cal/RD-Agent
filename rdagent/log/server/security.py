import hmac
import re
from pathlib import Path

from flask import (
    Response,
    current_app,
    jsonify,
    make_response,
    redirect,
    request,
    send_from_directory,
    url_for,
)

_PUBLIC_ENDPOINTS = {"favicon", "index", "server_static_files", "static"}


def serve_index() -> Response:
    token = current_app.config.get("AUTH_TOKEN", "")
    supplied_token = request.args.get("token", "")
    if token and supplied_token and hmac.compare_digest(supplied_token, token):
        response = make_response(redirect(url_for("index")))
        response.set_cookie("rdagent_auth", token, httponly=True, samesite="Strict")
        return response
    return send_from_directory(current_app.static_folder, "index.html")


def require_authentication() -> Response | tuple[Response, int] | None:
    token = current_app.config.get("AUTH_TOKEN", "")
    if not token or request.method == "OPTIONS" or request.endpoint in _PUBLIC_ENDPOINTS:
        return None

    authorization = request.headers.get("Authorization", "")
    header_token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
    provided_token = header_token or request.cookies.get("rdagent_auth", "")
    if not provided_token or not hmac.compare_digest(provided_token, token):
        return jsonify({"error": "Authentication required"}), 401
    return None


SCENARIO_TARGETS = {
    "Finance Data Building": "fin_factor",
    "Finance Model Implementation": "fin_model",
    "Finance Whole Pipeline": "fin_quant",
    "Finance Data Building (Reports)": "fin_factor_report",
    "General Model Implementation": "general_model",
    "Data Science": "data_science",
}

_COMPETITION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
_UNSAFE_UPLOAD_SUFFIXES = {".dill", ".pickle", ".pkl", ".py", ".pyc", ".pyo"}
_ERR_COMPETITION_PREFIX = "Competition must start with 'MLE-Bench:'"
_ERR_INVALID_COMPETITION = "Invalid competition name"
_ERR_INVALID_FILENAME = "Invalid upload filename"
_ERR_PATH_ESCAPE = "Path escapes the configured root"
_ERR_UNKNOWN_SCENARIO = "Unknown scenario"
_ERR_UNSAFE_FILE_TYPE = "Unsafe upload file type"


def validate_scenario(value: str | None) -> str:
    if value not in SCENARIO_TARGETS:
        raise ValueError(_ERR_UNKNOWN_SCENARIO)
    return value


def parse_competition(value: str | None) -> str:
    prefix = "MLE-Bench:"
    if value is None or not value.startswith(prefix):
        raise ValueError(_ERR_COMPETITION_PREFIX)
    competition = value[len(prefix) :]
    if not _COMPETITION_RE.fullmatch(competition):
        raise ValueError(_ERR_INVALID_COMPETITION)
    return competition


def resolve_within(root: str | Path, *parts: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved_path = resolved_root.joinpath(*parts).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(_ERR_PATH_ESCAPE) from exc
    return resolved_path


def validate_upload_filename(value: str) -> str:
    filename = Path(value).name
    if not filename or filename in {".", ".."}:
        raise ValueError(_ERR_INVALID_FILENAME)
    if Path(filename).suffix.lower() in _UNSAFE_UPLOAD_SUFFIXES:
        raise ValueError(_ERR_UNSAFE_FILE_TYPE)
    return filename
