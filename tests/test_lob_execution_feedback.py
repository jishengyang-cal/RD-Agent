import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from rdagent.app.lob_execution_feedback import (
    MIN_ROUND_TRIPS_PER_DAY,
    load_replay_feedback,
    rank_execution_candidates,
    score_replay_feedback,
    screen_execution_candidates,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_id(run_id: str, symbol: str, ts_recv_ns: int, horizon_ms: int) -> str:
    raw = f"{run_id}|{symbol}|{ts_recv_ns}|{horizon_ms}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _replay(
    tmp_path: Path,
    *,
    candidate_run_id: str,
    trading_date: str,
    entry_price: float,
    exit_price: float,
) -> Path:
    root = tmp_path / f"{candidate_run_id}-{trading_date}"
    root.mkdir()
    source = root / "source-manifest.json"
    source.write_text("{}", encoding="utf-8")
    catalog = root / "catalog"
    catalog.mkdir()
    catalog_receipt = catalog / "strict-l2-catalog-receipt.json"
    catalog_receipt.write_text(json.dumps({
        "schema_version": "strict-l2-nautilus-catalog/v2",
        "availability_tie_break_ns": 1,
        "symbol": "NVDA",
    }), encoding="utf-8")
    model = root / "model.pt"
    model.write_bytes(b"sealed model")
    audit = root / "candidate-audit.json"
    audit.write_text(json.dumps({
        "schema_version": "lob-candidate-audit/v1",
        "artifact_valid": True,
        "strict_l2_only": True,
        "run_id": candidate_run_id,
        "candidate_path": str(root),
        "model_sha256": _sha256(model),
        "symbols": ["NVDA"],
        "baseline_screen": {
            "screening_effective": True,
            "required_symbols": 1,
            "overall": {"horizons": {"1000ms": {"joint_baseline_win": True}}},
            "symbols": {
                "NVDA": {"horizons": {"1000ms": {"joint_baseline_win": True}}},
            },
        },
    }), encoding="utf-8")
    base = int(
        pd.Timestamp(f"{trading_date} 09:31:00", tz="America/New_York")
        .tz_convert("UTC")
        .value,
    )
    records = []
    for index in range(20):
        signal_ts = base + index * 2_000_000_000
        parent = _prediction_id(candidate_run_id, "NVDA", signal_ts, 1_000)
        records.extend(({
            "prediction_id": parent,
            "parent_prediction_id": parent,
            "instrument_uid": "NVDA.XNAS",
            "research_symbol": "NVDA",
            "signal_ts_recv_ns": signal_ts,
            "horizon_ms": 1_000,
            "action_role": "ENTRY",
            "side": "BUY",
            "status": "FILLED",
            "decision_ts_ns": signal_ts,
            "submit_ts_ns": signal_ts + 1,
            "last_fill_ts_ns": signal_ts + 2,
            "filled_qty": 10.0,
            "average_fill_price": entry_price,
            "fees": 0.1,
        }, {
            "prediction_id": f"{parent}-exit",
            "parent_prediction_id": parent,
            "instrument_uid": "NVDA.XNAS",
            "research_symbol": "NVDA",
            "signal_ts_recv_ns": signal_ts,
            "horizon_ms": 1_000,
            "action_role": "EXIT",
            "side": "SELL",
            "status": "FILLED",
            "decision_ts_ns": signal_ts + 1_000_000_000,
            "submit_ts_ns": signal_ts + 1_000_000_001,
            "last_fill_ts_ns": signal_ts + 1_000_000_002,
            "filled_qty": 10.0,
            "average_fill_price": exit_price,
            "fees": 0.1,
        }))
    feedback = {
        "schema_version": "research/execution-feedback-v1",
        "environment": "backtest",
        "trading_date": trading_date,
        "run_id": root.name,
        "model_id": candidate_run_id,
        "model_artifact_sha256": _sha256(model),
        "feature_manifest_sha256": _sha256(source),
        "reconciliation": {"status": "complete"},
        "metrics": {
            "execution_assumptions": {
                "execution_mode": "aggressive_marketable_fok",
                "market_data_availability": {
                    "research_boundary": "left_closed_ts_recv",
                    "catalog_ts_init_offset_ns": 1,
                },
                "fee_scenario": {
                    "model": "per_share",
                    "currency": "USD",
                    "fee_per_share": "0.01",
                },
                "latency_scenario": {
                    "model": "static_order_latency",
                    "unit": "nanosecond",
                    "order_insert_latency_ns": 1_000_000,
                },
            },
        },
        "records": records,
    }
    feedback_path = root / "execution-feedback.json"
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    result = {
        "schema_version": "strict-l2-candidate-replay-result/v1",
        "replay_id": root.name,
        "candidate_run_id": candidate_run_id,
        "trading_date": trading_date,
        "audit_receipt_sha256": _sha256(audit),
        "book_type": "L2_MBP",
        "execution_mode": "aggressive_marketable_fok",
        "market_data_availability": {
            "research_boundary": "left_closed_ts_recv",
            "catalog_ts_init_offset_ns": 1,
        },
        "trade_execution": True,
        "liquidity_consumption": True,
        "queue_position": True,
        "queue_semantics": "aggregate-level approximation, not L3 FIFO ground truth",
        "catalog_receipts": [{
            "symbol": "NVDA",
            "path": str(catalog_receipt),
            "sha256": _sha256(catalog_receipt),
        }],
        "fee_scenario": {
            "model": "per_share",
            "currency": "USD",
            "fee_per_share": "0.01",
        },
        "latency_scenario": {
            "model": "static_order_latency",
            "unit": "nanosecond",
            "order_insert_latency_ns": 1_000_000,
        },
        "feedback_file": feedback_path.name,
        "feedback_sha256": _sha256(feedback_path),
        "request": {
            "audit_receipt_path": str(audit),
            "source_manifest_path": str(source),
            "trading_date": trading_date,
            "horizon_ms": 1_000,
            "catalogs": [{"symbol": "NVDA", "catalog_path": str(catalog)}],
        },
    }
    result_path = root / "replay-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def test_score_replay_feedback_reconstructs_after_fee_round_trip(tmp_path: Path) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )

    score = score_replay_feedback([replay], candidate_run_id="candidate-a")

    assert score["round_trips"] == MIN_ROUND_TRIPS_PER_DAY
    assert score["fill_rate"] == 1.0
    assert score["net_pnl"] == pytest.approx(16.0)
    assert score["net_pnl_per_share"] == pytest.approx(0.08)
    assert score["execution_effective"] is False


def test_score_replay_feedback_counts_unfilled_fok_entry_as_fill_outcome(
    tmp_path: Path,
) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    result = json.loads(replay.read_text())
    feedback_path = replay.parent / result["feedback_file"]
    feedback = json.loads(feedback_path.read_text())
    parent = feedback["records"][0]["parent_prediction_id"]
    feedback["records"] = [
        record
        for record in feedback["records"]
        if record["parent_prediction_id"] != parent or record["action_role"] != "EXIT"
    ]
    missed = feedback["records"][0]
    missed.update(
        status="EXPIRED",
        last_fill_ts_ns=None,
        filled_qty=0.0,
        average_fill_price=None,
        fees=0.0,
    )
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    result["feedback_sha256"] = _sha256(feedback_path)
    replay.write_text(json.dumps(result), encoding="utf-8")

    score = score_replay_feedback([replay], candidate_run_id="candidate-a")

    assert score["entry_attempts"] == 20  # noqa: PLR2004
    assert score["filled_entries"] == 19  # noqa: PLR2004
    assert score["fill_rate"] == pytest.approx(0.95)
    assert score["round_trips"] == 19  # noqa: PLR2004


def test_score_replay_feedback_preserves_zero_trade_candidate(tmp_path: Path) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    result = json.loads(replay.read_text())
    feedback_path = replay.parent / result["feedback_file"]
    feedback = json.loads(feedback_path.read_text())
    feedback["records"] = []
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    result["feedback_sha256"] = _sha256(feedback_path)
    replay.write_text(json.dumps(result), encoding="utf-8")

    score = score_replay_feedback([replay], candidate_run_id="candidate-a")

    assert score["round_trips"] == 0
    assert score["entry_attempts"] == 0
    assert score["fill_rate"] == 0.0
    assert score["net_pnl"] == 0.0
    assert score["execution_effective"] is False


def test_replay_feedback_rejects_identity_leakage(tmp_path: Path) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    result = json.loads(replay.read_text())
    feedback_path = replay.parent / result["feedback_file"]
    feedback = json.loads(feedback_path.read_text())
    feedback["records"][0]["client_order_id"] = "forbidden"
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    result["feedback_sha256"] = _sha256(feedback_path)
    replay.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="execution identity"):
        load_replay_feedback(replay, expected_candidate_run_id="candidate-a")


def test_replay_feedback_rejects_changed_candidate_model(tmp_path: Path) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    (replay.parent / "model.pt").write_bytes(b"changed after audit")

    with pytest.raises(ValueError, match="model artifact is missing or changed"):
        load_replay_feedback(replay, expected_candidate_run_id="candidate-a")


def test_replay_feedback_rejects_changed_catalog_receipt(tmp_path: Path) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    result = json.loads(replay.read_text())
    receipt = Path(result["catalog_receipts"][0]["path"])
    value = json.loads(receipt.read_text())
    value["availability_tie_break_ns"] = 0
    receipt.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="catalog receipt digest mismatch"):
        load_replay_feedback(replay, expected_candidate_run_id="candidate-a")


def test_replay_feedback_rejects_wrong_prediction_parent(tmp_path: Path) -> None:
    replay = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    result = json.loads(replay.read_text())
    feedback_path = replay.parent / result["feedback_file"]
    feedback = json.loads(feedback_path.read_text())
    for record in feedback["records"]:
        record["parent_prediction_id"] = "wrong-parent"
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    result["feedback_sha256"] = _sha256(feedback_path)
    replay.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="prediction identity"):
        score_replay_feedback([replay], candidate_run_id="candidate-a")


def test_execution_registry_uses_complete_common_day_after_cost_ranking(tmp_path: Path) -> None:
    first = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.1,
    )
    second = _replay(
        tmp_path,
        candidate_run_id="candidate-b",
        trading_date="2026-05-11",
        entry_price=100.0,
        exit_price=100.05,
    )
    first_next_day = _replay(
        tmp_path,
        candidate_run_id="candidate-a",
        trading_date="2026-05-12",
        entry_price=100.0,
        exit_price=100.1,
    )
    second_next_day = _replay(
        tmp_path,
        candidate_run_id="candidate-b",
        trading_date="2026-05-12",
        entry_price=100.0,
        exit_price=100.05,
    )
    registry = tmp_path / "execution-screen.json"
    registry.write_text(json.dumps({
        "schema_version": "lob-execution-screen-registry/v1",
        "status": "ready",
        "eligible": [{"run_id": "candidate-a"}, {"run_id": "candidate-b"}],
    }), encoding="utf-8")
    output = tmp_path / "execution-registry.json"

    result = rank_execution_candidates(
        registry,
        {
            "candidate-a": [first, first_next_day],
            "candidate-b": [second, second_next_day],
        },
        output,
    )

    assert result["status"] == "complete"
    assert result["champion"]["candidate_run_id"] == "candidate-a"
    assert result["candidates"][1]["execution_effective"] is True
    assert output.is_file()


def test_execution_screen_audits_complete_shortlist_before_replay(tmp_path: Path) -> None:
    registry = tmp_path / "candidate-registry.json"
    registry.write_text(json.dumps({
        "schema_version": "lob-challenger-registry/v1",
        "shortlist": [{"run_id": "candidate-a"}, {"run_id": "candidate-b"}],
        "candidates": [
            {"run_id": "candidate-a"},
            {"run_id": "candidate-b"},
            {"run_id": "candidate-c"},
        ],
    }), encoding="utf-8")
    receipts = {}
    for run_id, passed in (
        ("candidate-a", True),
        ("candidate-b", False),
        ("candidate-c", True),
    ):
        candidate = tmp_path / run_id
        candidate.mkdir()
        model = candidate / "model.pt"
        model.write_bytes(run_id.encode())
        receipt = tmp_path / f"{run_id}-audit.json"
        receipt.write_text(json.dumps({
            "schema_version": "lob-candidate-audit/v1",
            "artifact_valid": True,
            "strict_l2_only": True,
            "run_id": run_id,
            "candidate_path": str(candidate),
            "model_sha256": _sha256(model),
            "baseline_screen": {
                "screening_effective": passed,
                "required_symbols": 1,
                "overall": {
                    "horizons": {"1000ms": {"joint_baseline_win": passed}},
                },
                "symbols": {
                    "NVDA": {
                        "horizons": {"1000ms": {"joint_baseline_win": passed}},
                    },
                },
            },
        }), encoding="utf-8")
        receipts[run_id] = receipt

    output = tmp_path / "execution-screen.json"
    result = screen_execution_candidates(
        registry,
        receipts,
        horizon_ms=1_000,
        output_path=output,
    )

    assert result["status"] == "ready"
    assert [item["run_id"] for item in result["eligible"]] == [
        "candidate-a",
        "candidate-c",
    ]
    assert result["audited_count"] == 3  # noqa: PLR2004
    assert result["replay_budget"] == 2  # noqa: PLR2004
    assert result["rejected"] == [{
        "run_id": "candidate-b",
        "reason": "requested_horizon_predictive_baseline_gate_failed",
    }]
    assert output.is_file()
