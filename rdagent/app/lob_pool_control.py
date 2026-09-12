"""Inspect strict-L2 candidate pools and create immutable recovery definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdagent.app.lob_model_loop import LOB_POOL_SCHEMA, _load_candidate_result, _run_id, validate_lob_pool


def inspect_lob_pool(pool_path: str | Path) -> dict[str, Any]:
    """Return validated candidate publication states without launching training."""
    definition, candidates = validate_lob_pool(pool_path)
    output_root = Path(str(candidates[0][1]["output_root"])).expanduser().resolve()
    states = []
    counts = {"complete": 0, "incomplete": 0, "pending": 0, "invalid": 0}
    for candidate_path, spec in candidates:
        run_id = _run_id(spec)
        final = output_root / run_id
        staging = output_root / f".incomplete-{run_id}"
        detail: dict[str, Any] = {
            "architecture": spec["architecture"],
            "variant": spec["variant"],
            "run_id": run_id,
            "source_spec": str(candidate_path),
        }
        if final.exists():
            try:
                result = _load_candidate_result(spec, candidate_path)
            except (OSError, TypeError, ValueError) as e:
                state = "invalid"
                detail["error_type"] = type(e).__name__
            else:
                state = "complete"
                detail["ranking"] = result["ranking"]
        elif staging.exists():
            state = "incomplete"
        else:
            state = "pending"
        counts[state] += 1
        detail["state"] = state
        states.append(detail)
    registry = output_root / f"challenger-pool-{definition['pool_id']}.json"
    return {
        "schema_version": "lob-pool-status/v1",
        "pool_id": definition["pool_id"],
        "registry_exists": registry.is_file(),
        "counts": counts,
        "candidates": states,
    }


def create_recovery_pool(
    pool_path: str | Path,
    output_path: str | Path,
    *,
    pool_id: str,
) -> dict[str, Any]:
    """Clone a validated pool under a new identity so completed runs can be reused."""
    definition, _ = validate_lob_pool(pool_path)
    target = Path(output_path).expanduser().resolve()
    staging = target.with_name(f".{target.name}.incomplete")
    if target.exists() or staging.exists():
        message = "recovery pool publication never overwrites output"
        raise FileExistsError(message)
    payload = {
        "schema_version": LOB_POOL_SCHEMA,
        "pool_id": pool_id,
        "candidates": definition["candidates"],
        "top_k": definition["top_k"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_lob_pool(staging)
        staging.replace(target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return {"status": "complete", "path": str(target), "pool_id": pool_id}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Inspect a candidate pool")
    status.add_argument("--pool", required=True)
    recovery = commands.add_parser("recovery", help="Create an immutable recovery pool")
    recovery.add_argument("--pool", required=True)
    recovery.add_argument("--output", required=True)
    recovery.add_argument("--pool-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pool inspection or recovery command."""
    args = _parser().parse_args(argv)
    if args.command == "status":
        result = inspect_lob_pool(args.pool)
    else:
        result = create_recovery_pool(args.pool, args.output, pool_id=args.pool_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
