"""Authenticate Nautilus strict-L2 replay feedback and rank execution candidates."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any
from zoneinfo import ZoneInfo

REPLAY_SCHEMA = "strict-l2-candidate-replay-result/v1"
FEEDBACK_SCHEMA = "research/execution-feedback-v1"
REGISTRY_SCHEMA = "lob-challenger-registry/v1"
EXECUTION_REGISTRY_SCHEMA = "lob-execution-registry/v1"
EXECUTION_SCREEN_SCHEMA = "lob-execution-screen-registry/v1"
FORBIDDEN_FIELDS = frozenset({
    "accountid",
    "brokerorderid",
    "clientorderid",
    "orderid",
    "venueorderid",
    "mpid",
})
TRADING_TIMEZONE = ZoneInfo("America/New_York")
MIN_EXECUTION_DAYS = 2
MIN_ROUND_TRIPS_PER_DAY = 20
MIN_ROUND_TRIPS_PER_SYMBOL = 10


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_execution_identities(value: object, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in FORBIDDEN_FIELDS:
                raise ValueError(f"{location} leaks an execution identity")
            _reject_execution_identities(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_execution_identities(child, f"{location}[{index}]")


def _finite_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be finite numeric data")
    return float(value)


def _timestamp_ns(value: object, field: str, trading_day: date) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative Unix nanosecond integer")
    local_day = datetime.fromtimestamp(
        value // 1_000_000_000,
        UTC,
    ).astimezone(TRADING_TIMEZONE).date()
    if local_day != trading_day:
        raise ValueError(f"{field} does not belong to the feedback trading date")
    return value


def _prediction_id(run_id: str, symbol: str, ts_recv_ns: int, horizon_ms: int) -> str:
    raw = f"{run_id}|{symbol}|{ts_recv_ns}|{horizon_ms}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _validated_fee_scenario(value: object) -> float:
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "currency", "fee_per_share"}
        or value.get("model") != "per_share"
        or value.get("currency") != "USD"
        or not isinstance(value.get("fee_per_share"), str)
    ):
        raise ValueError("replay result does not bind a per-share USD fee scenario")
    try:
        fee_per_share = float(value["fee_per_share"])
    except ValueError as exc:
        raise ValueError("replay fee scenario is not numeric") from exc
    if not math.isfinite(fee_per_share) or fee_per_share < 0:
        raise ValueError("replay fee scenario must be finite and non-negative")
    return fee_per_share


def _validated_latency_scenario(value: object) -> int:
    if (
        not isinstance(value, dict)
        or set(value) != {"model", "unit", "order_insert_latency_ns"}
        or value.get("model") != "static_order_latency"
        or value.get("unit") != "nanosecond"
    ):
        raise ValueError("replay result does not bind a static nanosecond latency scenario")
    latency = value["order_insert_latency_ns"]
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        raise ValueError("replay latency scenario must be a non-negative integer")
    return latency


def _validate_catalog_receipts(result: dict[str, Any], request: dict[str, Any]) -> None:
    """Re-authenticate the exact left-closed catalogs used by Nautilus."""
    catalogs = request.get("catalogs")
    bindings = result.get("catalog_receipts")
    if not isinstance(catalogs, list) or not isinstance(bindings, list) or len(catalogs) != len(bindings):
        raise ValueError("replay result has incomplete catalog receipt bindings")
    expected = {item.get("symbol"): item for item in catalogs if isinstance(item, dict)}
    if len(expected) != len(catalogs):
        raise ValueError("replay request has invalid catalog bindings")
    seen = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"symbol", "path", "sha256"}:
            raise ValueError("replay catalog receipt binding is invalid")
        symbol = binding["symbol"]
        requested = expected.get(symbol)
        if requested is None or symbol in seen:
            raise ValueError("replay catalog receipt symbol binding is invalid")
        seen.add(symbol)
        path = Path(str(binding["path"])).expanduser().resolve(strict=True)
        expected_path = (
            Path(str(requested.get("catalog_path"))).expanduser().resolve(strict=True)
            / "strict-l2-catalog-receipt.json"
        )
        if path != expected_path or _sha256(path) != binding["sha256"]:
            raise ValueError("replay catalog receipt digest mismatch")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != "strict-l2-nautilus-catalog/v2"
            or receipt.get("availability_tie_break_ns") != 1
            or receipt.get("symbol") != symbol
        ):
            raise ValueError("replay catalog does not preserve left-closed availability")


def _validate_audit_horizon_screen(audit: dict[str, Any], horizon_ms: object) -> None:
    if isinstance(horizon_ms, bool) or not isinstance(horizon_ms, int) or horizon_ms < 1:
        raise ValueError("replay request horizon must be a positive integer")
    screen = audit.get("baseline_screen")
    key = f"{horizon_ms}ms"
    if not isinstance(screen, dict):
        raise TypeError("candidate audit has no predictive baseline screen")
    symbols = screen.get("symbols")
    required = screen.get("required_symbols")
    overall = screen.get("overall")
    if (
        screen.get("screening_effective") is not True
        or not isinstance(overall, dict)
        or overall.get("horizons", {}).get(key, {}).get("joint_baseline_win") is not True
        or not isinstance(symbols, dict)
        or isinstance(required, bool)
        or not isinstance(required, int)
        or required < 1
    ):
        raise ValueError("candidate did not pass the replay-horizon baseline screen")
    wins = sum(
        isinstance(block, dict)
        and block.get("horizons", {}).get(key, {}).get("joint_baseline_win") is True
        for block in symbols.values()
    )
    if wins < required:
        raise ValueError("candidate baseline win does not generalize across enough symbols")


def load_replay_feedback(  # noqa: C901, PLR0912, PLR0915
    path: str | Path,
    *,
    expected_candidate_run_id: str,
) -> dict[str, Any]:
    """Authenticate one replay result and its identifier-free feedback file."""
    result_path = Path(path).expanduser().resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema_version") != REPLAY_SCHEMA:
        raise ValueError("invalid strict-L2 replay result")
    if result.get("candidate_run_id") != expected_candidate_run_id:
        raise ValueError("replay result belongs to a different candidate")
    if result.get("book_type") != "L2_MBP":
        raise ValueError("execution feedback must originate from L2_MBP replay")
    if (
        result.get("execution_mode") != "aggressive_marketable_fok"
        or result.get("trade_execution") is not True
        or result.get("liquidity_consumption") is not True
        or result.get("queue_position") is not True
    ):
        raise ValueError("replay result does not bind the audited execution configuration")
    if result.get("queue_semantics") != "aggregate-level approximation, not L3 FIFO ground truth":
        raise ValueError("replay result does not declare aggregate queue semantics")
    fee_per_share = _validated_fee_scenario(result.get("fee_scenario"))
    order_latency_ns = _validated_latency_scenario(result.get("latency_scenario"))
    feedback_name = result.get("feedback_file")
    if feedback_name != "execution-feedback.json":
        raise ValueError("replay feedback path is outside the immutable result contract")
    feedback_path = result_path.parent / feedback_name
    if not feedback_path.is_file() or _sha256(feedback_path) != result.get("feedback_sha256"):
        raise ValueError("replay feedback digest mismatch")
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    _reject_execution_identities(feedback)
    if not isinstance(feedback, dict) or feedback.get("schema_version") != FEEDBACK_SCHEMA:
        raise ValueError("invalid execution feedback schema")
    if (
        feedback.get("environment") != "backtest"
        or feedback.get("run_id") != result.get("replay_id")
        or feedback.get("model_id") != expected_candidate_run_id
        or feedback.get("reconciliation", {}).get("status") != "complete"
    ):
        raise ValueError("execution feedback identity or reconciliation mismatch")
    assumptions = feedback.get("metrics", {}).get("execution_assumptions")
    if assumptions != {
        "execution_mode": result["execution_mode"],
        "market_data_availability": result["market_data_availability"],
        "fee_scenario": result["fee_scenario"],
        "latency_scenario": result["latency_scenario"],
    }:
        raise ValueError("execution feedback assumptions differ from the replay result")
    request = result.get("request")
    if not isinstance(request, dict):
        raise TypeError("replay result has no bound request")
    if result.get("market_data_availability") != {
        "research_boundary": "left_closed_ts_recv",
        "catalog_ts_init_offset_ns": 1,
    }:
        raise ValueError("replay result does not bind left-closed market-data availability")
    _validate_catalog_receipts(result, request)
    audit_path = Path(str(request.get("audit_receipt_path"))).expanduser().resolve(strict=True)
    if _sha256(audit_path) != result.get("audit_receipt_sha256"):
        raise ValueError("candidate audit receipt digest mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != "lob-candidate-audit/v1"
        or audit.get("artifact_valid") is not True
        or audit.get("strict_l2_only") is not True
        or audit.get("run_id") != expected_candidate_run_id
        or feedback.get("model_artifact_sha256") != audit.get("model_sha256")
    ):
        raise ValueError("execution feedback is not bound to the audited strict-L2 candidate")
    audited_symbols = audit.get("symbols")
    if (
        not isinstance(audited_symbols, list)
        or not audited_symbols
        or len(audited_symbols) != len(set(audited_symbols))
        or not all(isinstance(symbol, str) and symbol for symbol in audited_symbols)
    ):
        raise ValueError("candidate audit has an invalid symbol set")
    _validate_audit_horizon_screen(audit, request.get("horizon_ms"))
    candidate = Path(str(audit.get("candidate_path"))).expanduser().resolve(strict=True)
    model = candidate / "model.pt"
    if (
        not candidate.is_dir()
        or model.is_symlink()
        or not model.is_file()
        or _sha256(model) != audit["model_sha256"]
    ):
        raise ValueError("audited candidate model artifact is missing or changed")
    source_manifest = Path(str(request.get("source_manifest_path"))).expanduser().resolve(strict=True)
    if feedback.get("feature_manifest_sha256") != _sha256(source_manifest):
        raise ValueError("execution feedback source manifest digest mismatch")
    records = feedback.get("records")
    if not isinstance(records, list):
        raise TypeError("execution feedback trade records must be a list")
    if (
        feedback.get("trading_date") != request.get("trading_date")
        or result.get("trading_date") != request.get("trading_date")
        or any(record.get("horizon_ms") != request["horizon_ms"] for record in records)
        or any(record.get("research_symbol") not in audited_symbols for record in records)
    ):
        raise ValueError("execution feedback differs from the replay date or horizon")
    feedback["_fee_per_share_usd"] = fee_per_share
    feedback["_order_insert_latency_ns"] = order_latency_ns
    feedback["_audited_symbols"] = tuple(sorted(audited_symbols))
    return feedback


def score_replay_feedback(  # noqa: C901, PLR0912, PLR0915
    replay_results: list[str | Path],
    *,
    candidate_run_id: str,
) -> dict[str, Any]:
    """Reconcile entry/exit records and calculate after-fee candidate economics."""
    if not replay_results:
        raise ValueError("at least one replay result is required")
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    trading_dates = set()
    fee_scenarios = set()
    latency_scenarios = set()
    audited_symbol_sets = set()
    prediction_ids = set()
    total_records = 0
    entry_attempts = 0
    filled_entries = 0
    for path in replay_results:
        feedback = load_replay_feedback(path, expected_candidate_run_id=candidate_run_id)
        fee_scenarios.add(feedback.pop("_fee_per_share_usd"))
        latency_scenarios.add(feedback.pop("_order_insert_latency_ns"))
        audited_symbol_sets.add(feedback.pop("_audited_symbols"))
        trading_date = feedback.get("trading_date")
        if not isinstance(trading_date, str) or trading_date in trading_dates:
            raise ValueError("candidate replay trading dates must be valid and unique")
        trading_day = date.fromisoformat(trading_date)
        trading_dates.add(trading_date)
        for index, record in enumerate(feedback["records"]):
            if not isinstance(record, dict):
                raise TypeError("execution feedback record must be an object")
            required = {
                "prediction_id",
                "parent_prediction_id",
                "instrument_uid",
                "signal_ts_recv_ns",
                "horizon_ms",
                "action_role",
                "research_symbol",
                "side",
                "status",
                "decision_ts_ns",
                "submit_ts_ns",
                "last_fill_ts_ns",
                "filled_qty",
                "average_fill_price",
                "fees",
            }
            if not required <= set(record):
                raise ValueError("execution feedback lacks causal round-trip fields")
            prediction_id = record["prediction_id"]
            if not isinstance(prediction_id, str) or not prediction_id or prediction_id in prediction_ids:
                raise ValueError("execution feedback prediction IDs must be non-empty and unique")
            prediction_ids.add(prediction_id)
            role = record["action_role"]
            parent = record["parent_prediction_id"]
            if (
                role not in {"ENTRY", "EXIT"}
                or not isinstance(parent, str)
                or not parent
                or not isinstance(record["research_symbol"], str)
                or not record["research_symbol"]
            ):
                raise ValueError("execution feedback round-trip identity is invalid")
            key = f"{trading_date}|{parent}"
            if role in pairs[key]:
                raise ValueError("execution feedback duplicates a round-trip action")
            signal_ts = _timestamp_ns(
                record["signal_ts_recv_ns"],
                f"records[{index}].signal_ts_recv_ns",
                trading_day,
            )
            decision_ts = _timestamp_ns(
                record["decision_ts_ns"],
                f"records[{index}].decision_ts_ns",
                trading_day,
            )
            submit_ts = _timestamp_ns(
                record["submit_ts_ns"],
                f"records[{index}].submit_ts_ns",
                trading_day,
            )
            filled_qty = _finite_number(record["filled_qty"], f"records[{index}].filled_qty")
            fill_ts = (
                None
                if record["last_fill_ts_ns"] is None
                else _timestamp_ns(
                    record["last_fill_ts_ns"],
                    f"records[{index}].last_fill_ts_ns",
                    trading_day,
                )
            )
            if not signal_ts <= decision_ts <= submit_ts or (
                fill_ts is not None and submit_ts > fill_ts
            ):
                raise ValueError("execution feedback violates causal timestamp ordering")
            if filled_qty > 0:
                if fill_ts is None or record["status"] not in {"FILLED", "MIXED"}:
                    raise ValueError("filled execution feedback has an invalid terminal outcome")
            elif (
                role != "ENTRY"
                or fill_ts is not None
                or record["average_fill_price"] is not None
                or record["status"] not in {"CANCELED", "EXPIRED"}
            ):
                raise ValueError("unfilled execution feedback must be a missed FOK entry")
            horizon = record["horizon_ms"]
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                raise ValueError("execution feedback horizon must be a positive integer")
            if _prediction_id(
                candidate_run_id,
                record["research_symbol"],
                signal_ts,
                horizon,
            ) != parent:
                raise ValueError("execution feedback prediction identity mismatch")
            pairs[key][role] = record
            total_records += 1
            if role == "ENTRY":
                entry_attempts += 1
                filled_entries += int(filled_qty > 0)
    if len(fee_scenarios) != 1:
        raise ValueError("candidate replay results must use one fee scenario")
    if len(latency_scenarios) != 1:
        raise ValueError("candidate replay results must use one latency scenario")
    if len(audited_symbol_sets) != 1:
        raise ValueError("candidate replay results disagree on audited symbols")
    trades = []
    for key, pair in sorted(pairs.items()):
        if set(pair) == {"ENTRY"} and pair["ENTRY"]["filled_qty"] == 0:
            continue
        if set(pair) != {"ENTRY", "EXIT"}:
            raise ValueError(f"execution feedback has an incomplete round trip: {key}")
        entry, exit_record = pair["ENTRY"], pair["EXIT"]
        if any(
            entry[field] != exit_record[field]
            for field in (
                "parent_prediction_id",
                "instrument_uid",
                "research_symbol",
                "signal_ts_recv_ns",
                "horizon_ms",
            )
        ):
            raise ValueError(f"execution feedback round-trip context mismatch: {key}")
        entry_qty = _finite_number(entry["filled_qty"], f"{key}.entry.filled_qty")
        exit_qty = _finite_number(exit_record["filled_qty"], f"{key}.exit.filled_qty")
        if entry_qty <= 0 or not math.isclose(entry_qty, exit_qty, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"execution feedback round-trip quantity mismatch: {key}")
        entry_price = _finite_number(entry["average_fill_price"], f"{key}.entry.price")
        exit_price = _finite_number(exit_record["average_fill_price"], f"{key}.exit.price")
        if entry_price <= 0 or exit_price <= 0:
            raise ValueError(f"execution feedback prices must be positive: {key}")
        total_fee = _finite_number(entry["fees"], f"{key}.entry.fees") + _finite_number(
            exit_record["fees"],
            f"{key}.exit.fees",
        )
        if total_fee < 0:
            raise ValueError(f"execution feedback fees cannot be negative: {key}")
        if exit_record["decision_ts_ns"] < entry["last_fill_ts_ns"]:
            raise ValueError(f"execution feedback exit precedes entry fill: {key}")
        if entry["side"] == "BUY" and exit_record["side"] == "SELL":
            gross = (exit_price - entry_price) * entry_qty
        elif entry["side"] == "SELL" and exit_record["side"] == "BUY":
            gross = (entry_price - exit_price) * entry_qty
        else:
            raise ValueError(f"execution feedback round-trip sides do not close: {key}")
        trades.append((
            key.split("|", 1)[0],
            entry["last_fill_ts_ns"],
            entry["research_symbol"],
            gross - total_fee,
            entry_qty,
            total_fee,
        ))
    trades.sort(key=lambda trade: (trade[0], trade[1]))
    pnl = [trade[3] for trade in trades]
    shares = [trade[4] for trade in trades]
    fees = [trade[5] for trade in trades]
    cumulative = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    for value in pnl:
        cumulative += value
        peak = max(peak, cumulative)
        maximum_drawdown = max(maximum_drawdown, peak - cumulative)
    deviation = pstdev(pnl) if len(pnl) > 1 else 0.0
    trade_sharpe = fmean(pnl) / deviation * math.sqrt(len(pnl)) if deviation > 0 else 0.0
    total_shares = sum(shares)
    net_pnl = sum(pnl)
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_shares: dict[str, float] = defaultdict(float)
    daily_trades: dict[str, int] = defaultdict(int)
    symbol_pnl: dict[str, float] = defaultdict(float)
    symbol_shares: dict[str, float] = defaultdict(float)
    symbol_trades: dict[str, int] = defaultdict(int)
    for trading_date, _, symbol, value, quantity, _ in trades:
        daily_pnl[trading_date] += value
        daily_shares[trading_date] += quantity
        daily_trades[trading_date] += 1
        symbol_pnl[symbol] += value
        symbol_shares[symbol] += quantity
        symbol_trades[symbol] += 1
    daily = {
        trading_date: {
            "round_trips": daily_trades[trading_date],
            "net_pnl": daily_pnl[trading_date],
            "net_pnl_per_share": daily_pnl[trading_date] / daily_shares[trading_date],
        }
        for trading_date in sorted(daily_pnl)
    }
    expected_symbols = next(iter(audited_symbol_sets))
    symbols = {
        symbol: {
            "round_trips": symbol_trades[symbol],
            "net_pnl": symbol_pnl[symbol],
            "net_pnl_per_share": (
                symbol_pnl[symbol] / symbol_shares[symbol]
                if symbol_shares[symbol] > 0
                else None
            ),
        }
        for symbol in expected_symbols
    }
    required_symbols = max(1, (len(symbols) + 1) // 2)
    effective_symbols = sum(
        value["round_trips"] >= MIN_ROUND_TRIPS_PER_SYMBOL and value["net_pnl"] > 0
        for value in symbols.values()
    )
    execution_effective = (
        len(daily) >= MIN_EXECUTION_DAYS
        and all(value["round_trips"] >= MIN_ROUND_TRIPS_PER_DAY for value in daily.values())
        and net_pnl > 0
        and all(value["net_pnl"] > 0 for value in daily.values())
        and effective_symbols >= required_symbols
    )
    return {
        "candidate_run_id": candidate_run_id,
        "trading_dates": sorted(trading_dates),
        "fee_per_share_usd": next(iter(fee_scenarios)),
        "order_insert_latency_ns": next(iter(latency_scenarios)),
        "round_trips": len(pnl),
        "records": total_records,
        "entry_attempts": entry_attempts,
        "filled_entries": filled_entries,
        "fill_rate": filled_entries / entry_attempts if entry_attempts else 0.0,
        "net_pnl": net_pnl,
        "net_pnl_per_share": net_pnl / total_shares if total_shares else 0.0,
        "pnl_per_trade": net_pnl / len(pnl) if pnl else 0.0,
        "trade_sharpe": trade_sharpe,
        "maximum_drawdown": maximum_drawdown,
        "fees": sum(fees),
        "win_rate": sum(value > 0 for value in pnl) / len(pnl) if pnl else 0.0,
        "daily": daily,
        "symbols": symbols,
        "effectiveness_gate": {
            "minimum_trading_days": MIN_EXECUTION_DAYS,
            "minimum_round_trips_per_day": MIN_ROUND_TRIPS_PER_DAY,
            "minimum_round_trips_per_symbol": MIN_ROUND_TRIPS_PER_SYMBOL,
            "required_profitable_symbols": required_symbols,
            "requires_positive_overall_and_each_day_after_fees": True,
        },
        "execution_effective": execution_effective,
    }


def _load_screen_audit(path: str | Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("candidate audit receipt must not be a symbolic link")
    audit_path = source.resolve(strict=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != "lob-candidate-audit/v1"
        or audit.get("artifact_valid") is not True
        or audit.get("strict_l2_only") is not True
        or audit.get("run_id") != run_id
    ):
        raise ValueError("shortlist audit identity or strict-L2 contract is invalid")
    candidate = Path(str(audit.get("candidate_path"))).expanduser().resolve(strict=True)
    model = candidate / "model.pt"
    if (
        candidate.name != run_id
        or not model.is_file()
        or model.is_symlink()
        or _sha256(model) != audit.get("model_sha256")
    ):
        raise ValueError("shortlist audit model binding is invalid")
    return audit_path, audit


def screen_execution_candidates(  # noqa: C901
    candidate_registry_path: str | Path,
    audit_receipts: dict[str, str | Path],
    *,
    horizon_ms: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Audit the full Qlib pool, then seal the bounded execution shortlist."""
    registry_path = Path(candidate_registry_path).expanduser().resolve(strict=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("invalid LOB challenger registry")
    shortlist = registry.get("shortlist")
    if not isinstance(shortlist, list) or not shortlist:
        raise ValueError("LOB challenger registry has no shortlist")
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < len(shortlist):
        raise ValueError("LOB challenger registry has no complete candidate pool")
    ordered = []
    for candidate in candidates:
        run_id = candidate.get("run_id") if isinstance(candidate, dict) else None
        if not isinstance(run_id, str) or not run_id or run_id in ordered:
            raise ValueError("LOB challenger registry candidate identities are invalid")
        ordered.append(run_id)
    expected = set(ordered)
    if None in expected or set(audit_receipts) != expected:
        raise ValueError("audit receipts must cover the complete Qlib candidate pool")
    passed = []
    rejected = []
    for run_id in ordered:
        audit_path, audit = _load_screen_audit(audit_receipts[run_id], run_id)
        try:
            _validate_audit_horizon_screen(audit, horizon_ms)
        except (TypeError, ValueError):
            rejected.append({
                "run_id": run_id,
                "reason": "requested_horizon_predictive_baseline_gate_failed",
            })
        else:
            passed.append({
                "run_id": run_id,
                "audit_receipt_path": str(audit_path),
                "audit_receipt_sha256": _sha256(audit_path),
            })
    replay_budget = len(shortlist)
    eligible = passed[:replay_budget]
    rejected.extend({
        "run_id": item["run_id"],
        "reason": "outside_precommitted_execution_replay_budget",
    } for item in passed[replay_budget:])
    payload = {
        "schema_version": EXECUTION_SCREEN_SCHEMA,
        "candidate_registry_sha256": _sha256(registry_path),
        "horizon_ms": horizon_ms,
        "audited_count": len(expected),
        "replay_budget": replay_budget,
        "eligible": eligible,
        "rejected": rejected,
        "status": "ready" if eligible else "no_effective_prediction_candidate",
    }
    target = Path(output_path).expanduser().resolve()
    staging = target.with_name(f".{target.name}.incomplete")
    if target.exists() or staging.exists():
        raise FileExistsError("execution screen registry never overwrites output")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(target)
    return payload


def rank_execution_candidates(
    execution_screen_path: str | Path,
    replay_results: dict[str, list[str | Path]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Rank the Qlib shortlist by authenticated Nautilus after-cost outcomes."""
    registry_path = Path(execution_screen_path).expanduser().resolve(strict=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != EXECUTION_SCREEN_SCHEMA
        or registry.get("status") != "ready"
    ):
        raise ValueError("invalid or ineligible LOB execution screen registry")
    shortlist = registry.get("eligible")
    if not isinstance(shortlist, list) or not shortlist:
        raise ValueError("LOB execution screen registry has no eligible candidates")
    expected = {candidate.get("run_id") for candidate in shortlist}
    if None in expected or set(replay_results) != expected:
        raise ValueError("Nautilus replay results must cover the complete Qlib shortlist")
    scores = [
        score_replay_feedback(replay_results[run_id], candidate_run_id=run_id)
        for run_id in sorted(expected)
    ]
    date_sets = {tuple(score["trading_dates"]) for score in scores}
    if len(date_sets) != 1:
        raise ValueError("execution candidates were not replayed on identical trading days")
    if len({score["fee_per_share_usd"] for score in scores}) != 1:
        raise ValueError("execution candidates were not replayed under the same fee scenario")
    if len({score["order_insert_latency_ns"] for score in scores}) != 1:
        raise ValueError("execution candidates were not replayed under the same latency scenario")
    ranked = sorted(
        scores,
        key=lambda score: (
            -score["net_pnl_per_share"],
            -score["trade_sharpe"],
            score["maximum_drawdown"],
            score["candidate_run_id"],
        ),
    )
    for rank, score in enumerate(ranked, start=1):
        score["rank"] = rank
    effective = [score for score in ranked if score["execution_effective"]]
    payload = {
        "schema_version": EXECUTION_REGISTRY_SCHEMA,
        "execution_screen_sha256": _sha256(registry_path),
        "ranking_rule": (
            "maximize after-fee net PnL/share, then trade Sharpe, then minimize drawdown"
        ),
        "trading_dates": list(next(iter(date_sets))),
        "candidates": ranked,
        "champion": effective[0] if effective else None,
        "status": "complete" if effective else "no_effective_candidate",
    }
    target = Path(output_path).expanduser().resolve()
    staging = target.with_name(f".{target.name}.incomplete")
    if target.exists() or staging.exists():
        raise FileExistsError("execution registry publication never overwrites output")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(target)
    return payload
