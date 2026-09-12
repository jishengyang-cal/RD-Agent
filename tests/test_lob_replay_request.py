from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdagent.app.lob_replay_request import build_replay_requests

EXPECTED_SCENARIOS = 2


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    manifest = _write(tmp_path / "source" / "manifest.json", {
        "schema_version": "research/published-dataset-manifest-v1",
        "dataset_kind": "strict-l2-mbp",
        "point_in_time": {"effective_at": "2026-05-11"},
    })
    readiness = _write(tmp_path / "readiness.json", {
        "schema_version": "strict-l2-training-readiness/v1",
        "formal_ready": True,
        "strict_l2_only": True,
        "publications": [{
            "source_manifest": str(manifest),
            "source_manifest_sha256": _sha256(manifest),
        }],
    })
    policy = _write(tmp_path / "policy.json", {
        "schema_version": "strict-l2-replay-policy/v1",
        "selection_status": "precommitted_before_test_candidate_publication",
        "symbols": ["NVDA", "TSLA"],
        "trading_dates": ["2026-05-11"],
        "signal_policy": {
            "horizon_ms": 1_000,
            "trade_size": "100",
            "min_abs_delta_ticks": 0.5,
            "min_direction_probability": 0.55,
            "cooldown_ms": 1_000,
            "max_signal_lag_ms": 0,
        },
        "execution_scenarios": [{
            "name": "base",
            "fee_per_share_usd": "0.005",
            "order_insert_latency_ns": 1_000_000,
        }, {
            "name": "stress",
            "fee_per_share_usd": "0.01",
            "order_insert_latency_ns": 5_000_000,
        }],
        "constraints": {
            "book_type": "L2_MBP",
            "execution_mode": "aggressive_marketable_fok",
            "test_threshold_retuning": False,
        },
    })
    audit = _write(tmp_path / "audit.json", {
        "schema_version": "lob-candidate-audit/v1",
        "run_id": "candidate-a",
        "artifact_valid": True,
        "strict_l2_only": True,
        "symbols": ["NVDA", "TSLA"],
        "baseline_screen": {
            "screening_effective": True,
            "required_symbols": 1,
            "overall": {
                "horizons": {"1000ms": {"joint_baseline_win": True}},
            },
            "symbols": {
                "NVDA": {
                    "horizons": {"1000ms": {"joint_baseline_win": True}},
                },
                "TSLA": {
                    "horizons": {"1000ms": {"joint_baseline_win": False}},
                },
            },
        },
    })
    screen = _write(tmp_path / "screen.json", {
        "schema_version": "lob-execution-screen-registry/v1",
        "status": "ready",
        "horizon_ms": 1_000,
        "eligible": [{
            "run_id": "candidate-a",
            "audit_receipt_path": str(audit),
            "audit_receipt_sha256": _sha256(audit),
        }],
    })
    catalog_root = tmp_path / "catalogs"
    for symbol in ("NVDA", "TSLA"):
        catalog = catalog_root / "2026-05-11" / symbol
        _write(catalog / "strict-l2-catalog-receipt.json", {
            "schema_version": "strict-l2-nautilus-catalog/v2",
            "availability_tie_break_ns": 1,
            "symbol": symbol,
            "source_manifest_sha256": _sha256(manifest),
            "catalog_path": str(catalog.resolve()),
            "instrument_id": f"{symbol}.XNAS",
        })
    return screen, policy, readiness, catalog_root


def test_build_replay_requests_seals_full_candidate_date_scenario_product(
    tmp_path: Path,
) -> None:
    screen, policy, readiness, catalogs = _inputs(tmp_path)
    output = tmp_path / "requests"

    result = build_replay_requests(
        screen,
        policy,
        readiness,
        catalogs,
        output,
        starting_balances=["1000000 USD"],
    )

    assert result["request_count"] == EXPECTED_SCENARIOS
    assert {item["scenario_name"] for item in result["requests"]} == {"base", "stress"}
    assert (output / "request-index.json").is_file()
    for entry in result["requests"]:
        path = Path(entry["path"])
        request = json.loads(path.read_text())
        assert entry["sha256"] == _sha256(path)
        assert request["schema_version"] == "strict-l2-candidate-replay-request/v1"
        assert request["catalogs"] == [{
            "symbol": "NVDA",
            "catalog_path": str((catalogs / "2026-05-11" / "NVDA").resolve()),
            "instrument_id": "NVDA.XNAS",
        }, {
            "symbol": "TSLA",
            "catalog_path": str((catalogs / "2026-05-11" / "TSLA").resolve()),
            "instrument_id": "TSLA.XNAS",
        }]


def test_build_replay_requests_rejects_old_catalog_availability(tmp_path: Path) -> None:
    screen, policy, readiness, catalogs = _inputs(tmp_path)
    receipt_path = catalogs / "2026-05-11" / "NVDA" / "strict-l2-catalog-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["availability_tie_break_ns"] = 0
    _write(receipt_path, receipt)

    with pytest.raises(ValueError, match="left-closed"):
        build_replay_requests(
            screen,
            policy,
            readiness,
            catalogs,
            tmp_path / "requests",
            starting_balances=["1000000 USD"],
        )

    assert not (tmp_path / "requests").exists()


def test_build_replay_requests_never_overwrites(tmp_path: Path) -> None:
    screen, policy, readiness, catalogs = _inputs(tmp_path)
    output = tmp_path / "requests"
    output.mkdir()

    with pytest.raises(FileExistsError, match="never overwrites"):
        build_replay_requests(
            screen,
            policy,
            readiness,
            catalogs,
            output,
            starting_balances=["1000000 USD"],
        )


def test_build_replay_requests_rechecks_horizon_eligibility(tmp_path: Path) -> None:
    screen, policy, readiness, catalogs = _inputs(tmp_path)
    screen_value = json.loads(screen.read_text())
    audit_path = Path(screen_value["eligible"][0]["audit_receipt_path"])
    audit = json.loads(audit_path.read_text())
    audit["baseline_screen"]["overall"]["horizons"]["1000ms"][
        "joint_baseline_win"
    ] = False
    _write(audit_path, audit)
    screen_value["eligible"][0]["audit_receipt_sha256"] = _sha256(audit_path)
    _write(screen, screen_value)

    with pytest.raises(ValueError, match="not eligible"):
        build_replay_requests(
            screen,
            policy,
            readiness,
            catalogs,
            tmp_path / "requests",
            starting_balances=["1000000 USD"],
        )
