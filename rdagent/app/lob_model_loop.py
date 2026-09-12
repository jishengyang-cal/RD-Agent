"""Fail-closed launcher for strict-L2 Qlib model experiments.

The LOB data and training implementation remain in the research repository.
RD-Agent owns candidate selection; this boundary only accepts the model/training
surface and never gives an agent write access to data, labels, or time splits.
"""

# The validator intentionally keeps every rejection in one auditable,
# fail-closed boundary. Subprocess arguments are a fixed argv assembled only
# after every path and field has passed that boundary.
# ruff: noqa: C901, EM101, EM102, S603, TRY003

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

LOB_EXPERIMENT_SCHEMA = "lob-experiment-spec/v1"
LOB_POOL_SCHEMA = "lob-challenger-pool/v1"
HORIZONS_MS = (250, 1000, 5000, 15000, 60000)
MAX_POOL_ID_LENGTH = 64
MAX_POOL_CANDIDATES = 20
ALLOWED_ARCHITECTURES = {
    "deeplob",
    "lstm",
    "mlp",
    "siamese_tlob",
    "wide_tlob",
}
ALLOWED_TOP_LEVEL = {
    "schema_version", "readiness_path", "window", "architecture", "variant",
    "segments", "seed", "dataset", "model", "output_root", "strict_l2_only",
    "use_historical_vap",
    "champion_lock",
}
ALLOWED_DATASET = {"context", "medium_context", "history_days", "stride"}
ALLOWED_MODEL = {
    "kwargs", "learning_rate", "epochs", "batch_size", "device", "early_stop",
    "gradient_clip", "num_workers", "prefetch_factor", "direction_loss_weight",
}
ARCHITECTURE_KWARGS = {
    "mlp": {"hidden", "use_symbol_embedding"},
    "lstm": {"hidden", "use_symbol_embedding"},
    "deeplob": {"hidden", "use_symbol_embedding", "patch_count"},
    "wide_tlob": {
        "side_dim", "heads", "spatial_layers", "temporal_layers", "dropout",
        "use_symbol_embedding", "spatial_latents",
    },
    "siamese_tlob": {
        "side_dim", "heads", "spatial_layers", "temporal_layers", "dropout",
        "use_symbol_embedding", "spatial_latents",
    },
}


def validate_lob_spec(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != ALLOWED_TOP_LEVEL:
        raise ValueError("LOB spec contains missing or agent-forbidden fields")
    if value["schema_version"] != LOB_EXPERIMENT_SCHEMA:
        raise ValueError("LOB experiment schema mismatch")
    if value["architecture"] not in ALLOWED_ARCHITECTURES:
        raise ValueError("architecture is outside the LOB challenger allowlist")
    if value["strict_l2_only"] is not True or value["use_historical_vap"] is not False:
        raise ValueError("RD-Agent LOB evolution is strict-L2-only")
    if value.get("window", {}).get("sealed_final") is not False:
        raise ValueError("RD-Agent must never observe the sealed final window")
    if value.get("champion_lock") is not None:
        raise ValueError("RD-Agent cannot receive a sealed champion lock")
    if set(value.get("segments", {})) != {"train", "valid", "test"}:
        raise ValueError("immutable train/valid/test segments are required")
    if set(value.get("dataset", {})) != ALLOWED_DATASET:
        raise ValueError("dataset search surface is not allowlisted")
    if set(value.get("model", {})) != ALLOWED_MODEL:
        raise ValueError("model search surface is not allowlisted")
    kwargs = value["model"]["kwargs"]
    if not isinstance(kwargs, dict) or not set(kwargs) <= ARCHITECTURE_KWARGS[value["architecture"]]:
        raise ValueError("architecture kwargs are outside the model-only allowlist")
    readiness = Path(value["readiness_path"]).expanduser().resolve(strict=True)
    readiness_value = json.loads(readiness.read_text(encoding="utf-8"))
    if (
        readiness_value.get("schema_version") != "strict-l2-training-readiness/v1"
        or readiness_value.get("formal_ready") is not True
        or readiness_value.get("strict_l2_only") is not True
    ):
        raise ValueError("verified strict-L2 readiness is required")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(spec: dict[str, object]) -> str:
    identity = dict(spec)
    identity["readiness_sha256"] = _sha256(str(spec["readiness_path"]))
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def _runner(qlib_python: str, research_root: str) -> tuple[Path, Path]:
    # Keep the virtualenv entrypoint path intact. Resolving its ``python``
    # symlink selects the base interpreter and silently drops the venv's
    # site-packages.
    python = Path(qlib_python).expanduser().absolute()
    root = Path(research_root).expanduser().resolve(strict=True)
    runner = root / "scripts" / "run_lob_experiment.py"
    if not python.is_file() or not runner.is_file():
        raise FileNotFoundError("Qlib Python or Research LOB runner is missing")
    return python, runner


def _audit_candidate(
    python: Path, research_root: Path, run_root: Path, spec_path: Path,
) -> dict[str, object]:
    audit_script = research_root.resolve(strict=True) / "scripts" / "audit_lob_candidate.py"
    if not audit_script.is_file():
        raise FileNotFoundError("Research LOB candidate auditor is missing")
    completed = subprocess.run(
        [
            str(python), str(audit_script), str(run_root),
            "--source-spec", str(spec_path),
        ],
        cwd=research_root,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout)
    if audit.get("artifact_valid") is not True:
        raise ValueError("LOB candidate did not pass the complete artifact audit")
    return audit


def _load_candidate_result(spec: dict[str, object], spec_path: Path) -> dict[str, object]:
    run_id = _run_id(spec)
    run_root = Path(str(spec["output_root"])).expanduser().resolve() / run_id
    required = {
        "checkpoint": run_root / "model.pt",
        "metrics": run_root / "metrics.json",
        "prediction_bundle": run_root / "prediction-bundle.json",
        "spec": run_root / "experiment-spec.json",
        "implementation": run_root / "implementation.json",
    }
    if not run_root.is_dir() or not all(path.is_file() for path in required.values()):
        raise ValueError("LOB candidate did not publish a complete immutable result")
    published_spec = json.loads(required["spec"].read_text(encoding="utf-8"))
    if published_spec != spec:
        raise ValueError("published LOB candidate spec differs from its requested spec")
    implementation = json.loads(required["implementation"].read_text(encoding="utf-8"))
    if (
        not isinstance(implementation, dict)
        or implementation.get("schema_version")
        not in {"lob-implementation/v1", "lob-implementation/v2"}
    ):
        raise ValueError("LOB candidate implementation manifest is invalid")
    recorded_implementation_sha = implementation.get("sha256")
    unsigned_implementation = dict(implementation)
    unsigned_implementation.pop("sha256", None)
    canonical = json.dumps(
        unsigned_implementation, sort_keys=True, separators=(",", ":"),
    ).encode()
    if recorded_implementation_sha != hashlib.sha256(canonical).hexdigest():
        raise ValueError("LOB candidate implementation digest mismatch")
    metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
    ranking = _ranking_metrics(metrics)
    return {
        "run_id": run_id,
        "architecture": spec["architecture"],
        "variant": spec["variant"],
        "source_spec": str(spec_path),
        "source_spec_sha256": _sha256(spec_path),
        "checkpoint_sha256": _sha256(required["checkpoint"]),
        "metrics_sha256": _sha256(required["metrics"]),
        "prediction_bundle_sha256": _sha256(required["prediction_bundle"]),
        "implementation_sha256": recorded_implementation_sha,
        "ranking": ranking,
    }


def _ranking_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, dict) or not isinstance(metrics.get("overall"), dict):
        raise TypeError("LOB metrics artifact has no overall metric block")
    overall = metrics["overall"]
    groups = {
        "mean_direction_f1_macro": [overall.get(f"direction_f1_macro_{h}ms") for h in HORIZONS_MS],
        "mean_up_brier": [overall.get(f"up_brier_{h}ms") for h in HORIZONS_MS],
        "mean_mae_ticks": [overall.get(f"mae_{h}ms") for h in HORIZONS_MS],
    }
    result = {}
    for name, values in groups.items():
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"LOB candidate lacks finite ranking metric {name}")
        result[name] = sum(values) / len(values)
    return result


def _ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, str]:
    ranking = candidate["ranking"]
    return (
        -ranking["mean_direction_f1_macro"],
        ranking["mean_up_brier"],
        ranking["mean_mae_ticks"],
        candidate["run_id"],
    )


def validate_lob_pool(path: str | Path) -> tuple[dict[str, object], list[tuple[Path, dict[str, object]]]]:
    source = Path(path).expanduser().resolve(strict=True)
    pool = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(pool, dict) or set(pool) != {"schema_version", "pool_id", "candidates", "top_k"}:
        raise ValueError("LOB pool fields do not match lob-challenger-pool/v1")
    if pool["schema_version"] != LOB_POOL_SCHEMA:
        raise ValueError("LOB challenger pool schema mismatch")
    pool_id = pool["pool_id"]
    if (
        not isinstance(pool_id, str)
        or not 1 <= len(pool_id) <= MAX_POOL_ID_LENGTH
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in pool_id)
    ):
        raise ValueError("LOB pool_id must use lowercase letters, digits, and hyphens")
    candidates = pool["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_POOL_CANDIDATES:
        raise ValueError("LOB challenger pool must contain 1..20 candidates")
    top_k = pool["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= min(5, len(candidates)):
        raise ValueError("LOB challenger top_k must be in 1..min(5, candidates)")
    loaded = []
    seen = set()
    comparison = None
    for candidate in candidates:
        candidate_path = Path(candidate).expanduser().resolve(strict=True)
        spec = validate_lob_spec(candidate_path)
        run_id = _run_id(spec)
        if run_id in seen:
            raise ValueError("LOB challenger pool contains a duplicate candidate run ID")
        seen.add(run_id)
        immutable = {
            name: spec[name]
            for name in ("readiness_path", "window", "segments", "output_root")
        }
        canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
        if comparison is None:
            comparison = canonical
        elif canonical != comparison:
            raise ValueError("LOB challenger candidates do not share the same evaluation cell")
        loaded.append((candidate_path, spec))
    return pool, loaded


def run_lob_pool(*, pool: str, qlib_python: str, research_root: str) -> dict[str, object]:
    definition, candidates = validate_lob_pool(pool)
    python, runner = _runner(qlib_python, research_root)
    output_root = Path(str(candidates[0][1]["output_root"])).expanduser().resolve()
    registry = output_root / f"challenger-pool-{definition['pool_id']}.json"
    staging = registry.with_name(f".{registry.name}.incomplete")
    if registry.exists() or staging.exists():
        raise FileExistsError("LOB challenger registry is immutable and already exists")
    completed = []
    failures = []
    for candidate_path, spec in candidates:
        run_root = output_root / _run_id(spec)
        try:
            if not run_root.exists():
                subprocess.run(
                    [
                        str(python),
                        str(runner),
                        "--spec",
                        str(candidate_path),
                        "--resume-incomplete",
                    ],
                    cwd=runner.parent.parent,
                    check=True,
                )
            result = _load_candidate_result(spec, candidate_path)
            result["audit"] = _audit_candidate(
                python, runner.parent.parent, run_root, candidate_path,
            )
            completed.append(result)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            failures.append({
                "source_spec": str(candidate_path),
                "source_spec_sha256": _sha256(candidate_path),
                "error_type": type(exc).__name__,
            })
    if not completed:
        raise RuntimeError("all LOB challenger candidates failed")
    if failures:
        raise RuntimeError(
            "LOB challenger pool is incomplete; recover failed candidates before ranking",
        )
    ranked = sorted(completed, key=_ranking_key)
    effective = [
        candidate for candidate in ranked
        if candidate["audit"].get("baseline_screen", {}).get("screening_effective") is True
    ]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    payload = {
        "schema_version": "lob-challenger-registry/v1",
        "pool_id": definition["pool_id"],
        "ranking_rule": (
            "maximize mean multi-horizon macro-F1, then minimize mean up-Brier, "
            "then minimize mean tick-MAE"
        ),
        "candidate_count": len(candidates),
        "completed_count": len(ranked),
        "failed_count": len(failures),
        "shortlist_rule": "complete audit and baseline_screen.screening_effective=true",
        "shortlist": effective[: definition["top_k"]],
        "candidates": ranked,
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(registry)
    return {"status": "complete", "registry": str(registry), **payload}
