"""Review complete base/stress Nautilus replay matrices for robust promotion."""

# ruff: noqa: C901, EM101, PLR0912, PLR0915, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rdagent.app.lob_execution_feedback import score_replay_feedback

INDEX_SCHEMA = "strict-l2-replay-request-index/v1"
RESULT_SCHEMA = "strict-l2-candidate-replay-result/v1"
REVIEW_SCHEMA = "strict-l2-robust-execution-review/v1"
MIN_EXECUTION_SCENARIOS = 2


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _expected_result_path(
    replay_root: Path,
    entry: dict[str, Any],
    request: dict[str, Any],
) -> Path:
    identity = {
        "request_sha256": entry["sha256"],
        "audit_receipt_sha256": request["audit_receipt_sha256"],
        "candidate_run_id": entry["candidate_run_id"],
        "trading_date": entry["trading_date"],
    }
    return replay_root / _canonical_sha256(identity)[:20] / "replay-result.json"


def review_execution_scenarios(
    request_index_path: str | Path,
    replay_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Require a complete identical-date matrix and select a stress-robust winner."""
    index_path = Path(request_index_path).expanduser().resolve(strict=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema_version") != INDEX_SCHEMA:
        raise ValueError("invalid strict-L2 replay request index")
    entries = index.get("requests")
    if not isinstance(entries, list) or not entries or index.get("request_count") != len(entries):
        raise ValueError("replay request index inventory is incomplete")
    root = Path(replay_root).expanduser().resolve(strict=True)
    grouped: dict[str, dict[str, list[Path]]] = {}
    expected_results: set[Path] = set()
    seen_cells: set[tuple[str, str, str]] = set()
    candidate_dates: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "candidate_run_id", "trading_date", "scenario_name", "path", "sha256",
        }:
            raise ValueError("replay request index entry is invalid")
        candidate = entry["candidate_run_id"]
        date = entry["trading_date"]
        scenario = entry["scenario_name"]
        cell = (candidate, date, scenario)
        if (
            not all(isinstance(value, str) and value for value in cell)
            or cell in seen_cells
        ):
            raise ValueError("replay request matrix contains an invalid or duplicate cell")
        seen_cells.add(cell)
        request_path = Path(entry["path"]).expanduser().resolve(strict=True)
        if _sha256(request_path) != entry["sha256"]:
            raise ValueError("replay request digest mismatch")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if (
            request.get("trading_date") != date
            or request.get("scenario_name") != scenario
            or request.get("audit_receipt_path") is None
        ):
            raise ValueError("replay request differs from its index cell")
        result_path = _expected_result_path(root, entry, request).resolve(strict=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("schema_version") != RESULT_SCHEMA
            or result.get("request_sha256") != entry["sha256"]
            or result.get("candidate_run_id") != candidate
            or result.get("trading_date") != date
            or result.get("request") != request
        ):
            raise ValueError("Nautilus result differs from its immutable request")
        expected_results.add(result_path)
        grouped.setdefault(scenario, {}).setdefault(candidate, []).append(result_path)
        candidate_dates.setdefault(candidate, set()).add(date)
    candidates = set(candidate_dates)
    dates = set.union(*candidate_dates.values())
    scenarios = set(grouped)
    if (
        len(scenarios) < MIN_EXECUTION_SCENARIOS
        or any(value != dates for value in candidate_dates.values())
        or any(set(values) != candidates for values in grouped.values())
        or len(entries) != len(candidates) * len(dates) * len(scenarios)
    ):
        raise ValueError("replay matrix must cover identical candidates, dates, and scenarios")
    actual_results = {
        path.resolve()
        for path in root.glob("*/replay-result.json")
        if path.is_file()
    }
    if actual_results != expected_results:
        raise ValueError("replay root contains missing or unindexed result publications")
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in sorted(scenarios):
        scores[scenario] = {}
        for candidate in sorted(candidates):
            score = score_replay_feedback(
                sorted(grouped[scenario][candidate]),
                candidate_run_id=candidate,
            )
            if score.get("trading_dates") != sorted(dates):
                raise ValueError("authenticated execution score uses different trading dates")
            scores[scenario][candidate] = score
    ranked = []
    for candidate in sorted(candidates):
        scenario_scores = {
            scenario: scores[scenario][candidate]
            for scenario in sorted(scenarios)
        }
        net_values = [value["net_pnl_per_share"] for value in scenario_scores.values()]
        sharpe_values = [value["trade_sharpe"] for value in scenario_scores.values()]
        drawdowns = [value["maximum_drawdown"] for value in scenario_scores.values()]
        if not all(math.isfinite(value) for value in (*net_values, *sharpe_values, *drawdowns)):
            raise ValueError("execution scenario score contains non-finite values")
        ranked.append({
            "candidate_run_id": candidate,
            "robust_effective": all(
                value.get("execution_effective") is True
                for value in scenario_scores.values()
            ),
            "worst_case_net_pnl_per_share": min(net_values),
            "worst_case_trade_sharpe": min(sharpe_values),
            "worst_case_maximum_drawdown": max(drawdowns),
            "scenarios": scenario_scores,
        })
    ranked.sort(key=lambda value: (
        -value["worst_case_net_pnl_per_share"],
        -value["worst_case_trade_sharpe"],
        value["worst_case_maximum_drawdown"],
        value["candidate_run_id"],
    ))
    for rank, value in enumerate(ranked, start=1):
        value["rank"] = rank
    effective = [value for value in ranked if value["robust_effective"]]
    payload = {
        "schema_version": REVIEW_SCHEMA,
        "request_index_path": str(index_path),
        "request_index_sha256": _sha256(index_path),
        "replay_root": str(root),
        "candidate_count": len(candidates),
        "trading_dates": sorted(dates),
        "scenarios": sorted(scenarios),
        "ranking_rule": (
            "maximize worst-scenario net PnL/share, then worst-scenario trade Sharpe, "
            "then minimize worst-scenario drawdown"
        ),
        "candidates": ranked,
        "champion": effective[0] if effective else None,
        "status": "complete" if effective else "no_robust_effective_candidate",
    }
    target = Path(output_path).expanduser().resolve()
    staging = target.with_name(f".{target.name}.incomplete")
    if target.exists() or staging.exists():
        raise FileExistsError("robust execution review never overwrites output")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(target)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-index", required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = review_execution_scenarios(args.request_index, args.replay_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
