from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdagent.app import lob_execution_review as review


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    replay_root = tmp_path / "replays"
    entries = []
    for candidate in ("candidate-a", "candidate-b"):
        for trading_date in ("2026-05-11", "2026-05-12"):
            for scenario in ("base", "stress"):
                request = {
                    "audit_receipt_path": str(tmp_path / f"{candidate}-audit.json"),
                    "audit_receipt_sha256": "a" * 64,
                    "trading_date": trading_date,
                    "scenario_name": scenario,
                }
                request_path = _write(
                    tmp_path / "requests" / candidate / trading_date / f"{scenario}.json",
                    request,
                )
                entry = {
                    "candidate_run_id": candidate,
                    "trading_date": trading_date,
                    "scenario_name": scenario,
                    "path": str(request_path),
                    "sha256": _sha256(request_path),
                }
                identity = {
                    "request_sha256": entry["sha256"],
                    "audit_receipt_sha256": request["audit_receipt_sha256"],
                    "candidate_run_id": candidate,
                    "trading_date": trading_date,
                }
                replay_id = _canonical_sha256(identity)[:20]
                _write(replay_root / replay_id / "replay-result.json", {
                    "schema_version": "strict-l2-candidate-replay-result/v1",
                    "request_sha256": entry["sha256"],
                    "candidate_run_id": candidate,
                    "trading_date": trading_date,
                    "request": request,
                })
                entries.append(entry)
    index = _write(tmp_path / "requests" / "request-index.json", {
        "schema_version": "strict-l2-replay-request-index/v1",
        "request_count": len(entries),
        "requests": entries,
    })
    return index, replay_root


def _fake_score(paths: list[Path], *, candidate_run_id: str) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in paths]
    scenario = results[0]["request"]["scenario_name"]
    assert {value["request"]["scenario_name"] for value in results} == {scenario}
    if candidate_run_id == "candidate-a":
        net, effective = (0.02 if scenario == "base" else 0.01), True
    else:
        net, effective = (0.04 if scenario == "base" else -0.01), scenario == "base"
    return {
        "candidate_run_id": candidate_run_id,
        "trading_dates": sorted(value["trading_date"] for value in results),
        "net_pnl_per_share": net,
        "trade_sharpe": net * 10,
        "maximum_drawdown": abs(net) / 2,
        "execution_effective": effective,
    }


def test_review_execution_scenarios_requires_stress_robust_champion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, replay_root = _fixture(tmp_path)
    monkeypatch.setattr(review, "score_replay_feedback", _fake_score)

    result = review.review_execution_scenarios(
        index,
        replay_root,
        tmp_path / "robust-review.json",
    )

    assert result["status"] == "complete"
    assert result["champion"]["candidate_run_id"] == "candidate-a"
    by_candidate = {value["candidate_run_id"]: value for value in result["candidates"]}
    assert by_candidate["candidate-a"]["robust_effective"] is True
    assert by_candidate["candidate-b"]["robust_effective"] is False


def test_review_execution_scenarios_rejects_unindexed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, replay_root = _fixture(tmp_path)
    _write(replay_root / "unexpected" / "replay-result.json", {})
    monkeypatch.setattr(review, "score_replay_feedback", _fake_score)

    with pytest.raises(ValueError, match="unindexed"):
        review.review_execution_scenarios(
            index,
            replay_root,
            tmp_path / "robust-review.json",
        )
