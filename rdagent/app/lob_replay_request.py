"""Build immutable Nautilus requests from an audited strict-L2 shortlist."""

# ruff: noqa: C901, EM101, PLR0912, PLR0915, TRY003, TRY301

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SCREEN_SCHEMA = "lob-execution-screen-registry/v1"
POLICY_SCHEMA = "strict-l2-replay-policy/v1"
READINESS_SCHEMA = "strict-l2-training-readiness/v1"
REQUEST_SCHEMA = "strict-l2-candidate-replay-request/v1"
INDEX_SCHEMA = "strict-l2-replay-request-index/v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("replay request inputs must not be symbolic links")
    source = source.resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("replay request input must be a JSON object")
    return source, value


def _source_manifests(readiness: dict[str, Any]) -> dict[str, Path]:
    if (
        readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("formal_ready") is not True
        or readiness.get("strict_l2_only") is not True
    ):
        raise ValueError("verified strict-L2 readiness is required")
    result: dict[str, Path] = {}
    publications = readiness.get("publications")
    if not isinstance(publications, list):
        raise TypeError("readiness publications must be a list")
    for publication in publications:
        if not isinstance(publication, dict):
            raise TypeError("readiness publication must be an object")
        source, manifest = _load(publication.get("source_manifest"))
        date = manifest.get("point_in_time", {}).get("effective_at")
        if (
            manifest.get("schema_version") != "research/published-dataset-manifest-v1"
            or manifest.get("dataset_kind") != "strict-l2-mbp"
            or not isinstance(date, str)
            or date in result
        ):
            raise ValueError("readiness has an invalid or duplicate strict-L2 source date")
        if _sha256(source) != publication.get("source_manifest_sha256"):
            raise ValueError("readiness source manifest digest mismatch")
        result[date] = source
    return result


def _catalogs(
    root: Path,
    *,
    trading_date: str,
    symbols: list[str],
    source_manifest: Path,
) -> list[dict[str, str]]:
    source_digest = _sha256(source_manifest)
    values = []
    for symbol in symbols:
        catalog = (root / trading_date / symbol).resolve(strict=True)
        catalog.relative_to(root)
        receipt_path = catalog / "strict-l2-catalog-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != "strict-l2-nautilus-catalog/v2"
            or receipt.get("availability_tie_break_ns") != 1
            or receipt.get("symbol") != symbol
            or receipt.get("source_manifest_sha256") != source_digest
            or Path(str(receipt.get("catalog_path"))).expanduser().resolve() != catalog
            or not isinstance(receipt.get("instrument_id"), str)
            or not receipt["instrument_id"]
        ):
            raise ValueError("catalog receipt is not the requested left-closed source binding")
        values.append({
            "symbol": symbol,
            "catalog_path": str(catalog),
            "instrument_id": receipt["instrument_id"],
        })
    return values


def _validate_audit_replay_eligibility(
    audit: dict[str, Any],
    *,
    horizon_ms: int,
    symbols: list[str],
) -> None:
    screen = audit.get("baseline_screen")
    key = f"{horizon_ms}ms"
    if (
        audit.get("symbols") != sorted(symbols)
        or not isinstance(screen, dict)
        or screen.get("screening_effective") is not True
        or screen.get("overall", {}).get("horizons", {}).get(key, {}).get(
            "joint_baseline_win",
        )
        is not True
    ):
        raise ValueError("candidate is not eligible at the replay horizon and symbol set")
    required = screen.get("required_symbols")
    per_symbol = screen.get("symbols")
    if (
        isinstance(required, bool)
        or not isinstance(required, int)
        or required < 1
        or not isinstance(per_symbol, dict)
        or set(per_symbol) != set(symbols)
        or sum(
            value.get("horizons", {}).get(key, {}).get("joint_baseline_win") is True
            for value in per_symbol.values()
            if isinstance(value, dict)
        )
        < required
    ):
        raise ValueError("candidate replay-horizon baseline win does not generalize")


def build_replay_requests(
    execution_screen_path: str | Path,
    replay_policy_path: str | Path,
    readiness_path: str | Path,
    catalog_root: str | Path,
    output_root: str | Path,
    *,
    starting_balances: list[str],
) -> dict[str, Any]:
    """Seal every candidate/date/scenario replay request and one inventory."""
    screen_path, screen = _load(execution_screen_path)
    policy_path, policy = _load(replay_policy_path)
    readiness_path, readiness = _load(readiness_path)
    if screen.get("schema_version") != SCREEN_SCHEMA or screen.get("status") != "ready":
        raise ValueError("execution screen has no eligible audited candidates")
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("selection_status")
        != "precommitted_before_test_candidate_publication"
        or policy.get("constraints", {}).get("book_type") != "L2_MBP"
        or policy.get("constraints", {}).get("execution_mode")
        != "aggressive_marketable_fok"
        or policy.get("constraints", {}).get("test_threshold_retuning") is not False
    ):
        raise ValueError("replay policy does not preserve the precommitted L2 screen")
    if (
        not isinstance(starting_balances, list)
        or not starting_balances
        or not all(isinstance(value, str) and value for value in starting_balances)
    ):
        raise ValueError("starting_balances must contain non-empty Nautilus money strings")
    symbols = policy.get("symbols")
    dates = policy.get("trading_dates")
    scenarios = policy.get("execution_scenarios")
    signal = policy.get("signal_policy")
    if (
        not isinstance(symbols, list)
        or not symbols
        or len(symbols) != len(set(symbols))
        or not all(isinstance(value, str) and value for value in symbols)
        or not isinstance(dates, list)
        or not dates
        or len(dates) != len(set(dates))
        or not isinstance(scenarios, list)
        or not scenarios
        or not isinstance(signal, dict)
    ):
        raise ValueError("replay policy dates, symbols, scenarios, or signal policy are invalid")
    scenario_names = [
        value.get("name") if isinstance(value, dict) else None
        for value in scenarios
    ]
    if (
        any(not isinstance(value, str) or not value for value in scenario_names)
        or len(scenario_names) != len(set(scenario_names))
    ):
        raise ValueError("execution scenario names must be non-empty and unique")
    if screen.get("horizon_ms") != signal.get("horizon_ms"):
        raise ValueError("execution screen and replay policy horizons differ")
    eligible = screen.get("eligible")
    if not isinstance(eligible, list) or not eligible:
        raise ValueError("execution screen eligible candidates are invalid")
    run_ids = [value.get("run_id") if isinstance(value, dict) else None for value in eligible]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("execution screen contains duplicate eligible candidates")
    source_by_date = _source_manifests(readiness)
    catalog_base = Path(catalog_root).expanduser()
    if catalog_base.is_symlink():
        raise ValueError("catalog root must not be a symbolic link")
    catalog_base = catalog_base.resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    index_path = output / "request-index.json"
    if output.exists() or index_path.exists():
        raise FileExistsError("replay request publication never overwrites output")
    requests: list[dict[str, Any]] = []
    created_output = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created_output = True
        for candidate in eligible:
            if not isinstance(candidate, dict):
                raise TypeError("eligible candidate must be an object")
            run_id = candidate.get("run_id")
            audit = Path(str(candidate.get("audit_receipt_path"))).expanduser().resolve(strict=True)
            audit_value = json.loads(audit.read_text(encoding="utf-8"))
            if (
                not isinstance(run_id, str)
                or not run_id
                or _sha256(audit) != candidate.get("audit_receipt_sha256")
                or not isinstance(audit_value, dict)
                or audit_value.get("schema_version") != "lob-candidate-audit/v1"
                or audit_value.get("artifact_valid") is not True
                or audit_value.get("strict_l2_only") is not True
                or audit_value.get("run_id") != run_id
            ):
                raise ValueError("eligible candidate audit binding is invalid")
            _validate_audit_replay_eligibility(
                audit_value,
                horizon_ms=signal["horizon_ms"],
                symbols=symbols,
            )
            for trading_date in dates:
                source_manifest = source_by_date.get(trading_date)
                if source_manifest is None:
                    raise ValueError("replay date is absent from verified readiness")
                catalogs = _catalogs(
                    catalog_base,
                    trading_date=trading_date,
                    symbols=symbols,
                    source_manifest=source_manifest,
                )
                for scenario in scenarios:
                    if not isinstance(scenario, dict) or set(scenario) != {
                        "name", "fee_per_share_usd", "order_insert_latency_ns",
                    }:
                        raise ValueError("execution scenario fields are invalid")
                    request = {
                        "schema_version": REQUEST_SCHEMA,
                        "audit_receipt_path": str(audit),
                        "audit_receipt_sha256": candidate["audit_receipt_sha256"],
                        "trading_date": trading_date,
                        "source_manifest_path": str(source_manifest),
                        "catalogs": catalogs,
                        "horizon_ms": signal["horizon_ms"],
                        "trade_size": signal["trade_size"],
                        "min_abs_delta_ticks": signal["min_abs_delta_ticks"],
                        "min_direction_probability": signal["min_direction_probability"],
                        "cooldown_ms": signal["cooldown_ms"],
                        "max_signal_lag_ms": signal["max_signal_lag_ms"],
                        "starting_balances": starting_balances,
                        "fee_per_share_usd": scenario["fee_per_share_usd"],
                        "order_insert_latency_ns": scenario["order_insert_latency_ns"],
                        "replay_policy_path": str(policy_path),
                        "replay_policy_sha256": _sha256(policy_path),
                        "scenario_name": scenario["name"],
                    }
                    path = output / run_id / trading_date / f"{scenario['name']}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(request, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    requests.append({
                        "candidate_run_id": run_id,
                        "trading_date": trading_date,
                        "scenario_name": scenario["name"],
                        "path": str(path),
                        "sha256": _sha256(path),
                    })
        payload = {
            "schema_version": INDEX_SCHEMA,
            "execution_screen_path": str(screen_path),
            "execution_screen_sha256": _sha256(screen_path),
            "replay_policy_path": str(policy_path),
            "replay_policy_sha256": _sha256(policy_path),
            "readiness_path": str(readiness_path),
            "readiness_sha256": _sha256(readiness_path),
            "catalog_root": str(catalog_base),
            "request_count": len(requests),
            "requests": requests,
        }
        index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        if created_output:
            shutil.rmtree(output, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-screen", required=True)
    parser.add_argument("--replay-policy", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--starting-balance", action="append", required=True)
    args = parser.parse_args()
    result = build_replay_requests(
        args.execution_screen,
        args.replay_policy,
        args.readiness,
        args.catalog_root,
        args.output_root,
        starting_balances=args.starting_balance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
