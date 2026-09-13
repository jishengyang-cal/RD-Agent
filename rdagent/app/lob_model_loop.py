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
    "schema_version",
    "readiness_path",
    "window",
    "architecture",
    "variant",
    "segments",
    "seed",
    "dataset",
    "model",
    "output_root",
    "strict_l2_only",
    "use_historical_vap",
    "champion_lock",
}
ALLOWED_DATASET = {"context", "medium_context", "history_days", "stride"}
ALLOWED_MODEL = {
    "kwargs",
    "learning_rate",
    "epochs",
    "batch_size",
    "device",
    "early_stop",
    "gradient_clip",
    "num_workers",
    "prefetch_factor",
    "direction_loss_weight",
}
ARCHITECTURE_KWARGS = {
    "mlp": {"hidden", "use_symbol_embedding"},
    "lstm": {"hidden", "use_symbol_embedding"},
    "deeplob": {"hidden", "use_symbol_embedding", "patch_count"},
    "wide_tlob": {
        "side_dim",
        "heads",
        "spatial_layers",
        "temporal_layers",
        "dropout",
        "use_symbol_embedding",
        "spatial_latents",
    },
    "siamese_tlob": {
        "side_dim",
        "heads",
        "spatial_layers",
        "temporal_layers",
        "dropout",
        "use_symbol_embedding",
        "spatial_latents",
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
    python: Path,
    research_root: Path,
    run_root: Path,
    spec_path: Path,
    *,
    tracking_uri: str | None = None,
) -> dict[str, object]:
    audit_script = research_root.resolve(strict=True) / "scripts" / "audit_lob_candidate.py"
    if not audit_script.is_file():
        raise FileNotFoundError("Research LOB candidate auditor is missing")
    arguments = [
        str(python),
        str(audit_script),
        str(run_root),
        "--source-spec",
        str(spec_path),
    ]
    if tracking_uri is not None:
        context_path = run_root.parent / "tracking" / run_root.name / "tracking-context.json"
        if any(path.is_symlink() for path in (context_path, context_path.parent, context_path.parent.parent)):
            raise ValueError("tracking context must not be a symbolic link")
        context = json.loads(context_path.read_text(encoding="utf-8"))
        attempts = context.get("attempts")
        if context.get("schema_version") != "lob-tracking-context/v1" or not isinstance(attempts, list) or not attempts:
            raise ValueError("training Recorder context is missing or malformed")
        attempt = attempts[-1]
        if not isinstance(attempt, dict) or (
            attempt.get("tracking_uri") != tracking_uri
            or attempt.get("run_id") != run_root.name
            or attempt.get("spec_sha256") != _sha256(spec_path)
            or not isinstance(attempt.get("recorder_id"), str)
            or not attempt["recorder_id"]
        ):
            raise ValueError("training Recorder context identity mismatch")
        arguments.extend(["--tracking-uri", tracking_uri, "--recorder-id", attempt["recorder_id"]])
    completed = subprocess.run(
        arguments,
        cwd=research_root,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout)
    if audit.get("artifact_valid") is not True:
        raise ValueError("LOB candidate did not pass the complete artifact audit")
    if audit.get("evaluation_segment") != "validation":
        raise ValueError("LOB candidate ranking requires an independent validation audit")
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
    if not isinstance(implementation, dict) or implementation.get("schema_version") not in {
        "lob-implementation/v1",
        "lob-implementation/v2",
    }:
        raise ValueError("LOB candidate implementation manifest is invalid")
    recorded_implementation_sha = implementation.get("sha256")
    unsigned_implementation = dict(implementation)
    unsigned_implementation.pop("sha256", None)
    canonical = json.dumps(
        unsigned_implementation,
        sort_keys=True,
        separators=(",", ":"),
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
        "evaluation_segment": metrics["evaluation_segment"],
        "ranking": ranking,
    }


def _ranking_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, dict) or not isinstance(metrics.get("overall"), dict):
        raise TypeError("LOB metrics artifact has no overall metric block")
    if metrics.get("evaluation_segment") != "validation":
        raise ValueError("LOB candidate ranking requires validation metrics, never test metrics")
    overall = metrics["overall"]
    groups = {
        "mean_direction_f1_macro": [overall.get(f"direction_f1_macro_{h}ms") for h in HORIZONS_MS],
        "mean_up_brier": [overall.get(f"up_brier_{h}ms") for h in HORIZONS_MS],
        "mean_mae_ticks": [overall.get(f"mae_{h}ms") for h in HORIZONS_MS],
    }
    result = {}
    for name, values in groups.items():
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
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
        immutable = {name: spec[name] for name in ("readiness_path", "window", "segments", "output_root")}
        canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
        if comparison is None:
            comparison = canonical
        elif canonical != comparison:
            raise ValueError("LOB challenger candidates do not share the same evaluation cell")
        loaded.append((candidate_path, spec))
    return pool, loaded


def inspect_lob_pool(*, pool: str, tracking_client: Any, experiment_ids: list[str]) -> dict[str, object]:
    """Observe all candidates via an existing MLflow client, without launching work.

    The caller owns backend/credentials and explicitly selects experiments.
    Recorded audit summaries are not a new audit or a promotion decision.
    """
    definition, candidates = validate_lob_pool(pool)
    if not experiment_ids or any(not isinstance(value, str) or not value for value in experiment_ids):
        raise ValueError("explicit nonempty experiment IDs are required")
    observed = []
    for spec_path, spec in candidates:
        business_id = _run_id(spec)
        spec_digest = _sha256(spec_path)
        attempts = []
        token = None
        seen_tokens = set()
        while True:
            page = tracking_client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=f"params.run_id = '{business_id}'",
                order_by=["attributes.start_time ASC", "attributes.run_id ASC"],
                max_results=100,
                page_token=token,
            )
            for run in page:
                identity_valid = (
                    run.data.params.get("run_id") == business_id
                    and run.data.params.get("spec_sha256") == spec_digest
                    and run.data.params.get("architecture") == spec["architecture"]
                    and run.data.params.get("sealed_final", "").lower() == "false"
                )
                attempts.append(
                    {
                        "recorder_id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "execution_status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "identity_valid": identity_valid,
                        "recorded_metrics": dict(run.data.metrics) if identity_valid else {},
                        "recorded_summary": (
                            {
                                key: value
                                for key, value in run.data.tags.items()
                                if key.startswith("audit.") or key == "training_stage"
                            }
                            if identity_valid
                            else {}
                        ),
                    }
                )
            token = page.token
            if not token:
                break
            if token in seen_tokens:
                raise ValueError("tracking pagination repeated a token")
            seen_tokens.add(token)
        observed.append(
            {
                "run_id": business_id,
                "architecture": spec["architecture"],
                "source_spec": str(spec_path),
                "source_spec_sha256": spec_digest,
                "configuration": spec,
                "tracking_state": "observed" if attempts else "unknown",
                "attempts": attempts,
            }
        )
    return {
        "schema_version": "lob-pool-observation/v1",
        "pool_id": definition["pool_id"],
        "experiment_ids": list(experiment_ids),
        "candidate_count": len(observed),
        "candidates": observed,
        "audit_evidence_reverified": False,
        "process_liveness_verified": False,
        "ranking_performed": False,
    }


def run_lob_pool(
    *, pool: str, qlib_python: str, research_root: str, tracking_uri: str | None = None
) -> dict[str, object]:
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
                python,
                runner.parent.parent,
                run_root,
                candidate_path,
                **({"tracking_uri": tracking_uri} if tracking_uri is not None else {}),
            )
            completed.append(result)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            failures.append(
                {
                    "source_spec": str(candidate_path),
                    "source_spec_sha256": _sha256(candidate_path),
                    "error_type": type(exc).__name__,
                },
            )
    if not completed:
        raise RuntimeError("all LOB challenger candidates failed")
    if failures:
        raise RuntimeError(
            "LOB challenger pool is incomplete; recover failed candidates before ranking",
        )
    ranked = sorted(completed, key=_ranking_key)
    effective = [
        candidate
        for candidate in ranked
        if candidate["audit"].get("baseline_screen", {}).get("screening_effective") is True
    ]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    payload = {
        "schema_version": "lob-challenger-registry/v1",
        "pool_id": definition["pool_id"],
        "ranking_rule": (
            "validation only: maximize mean multi-horizon macro-F1, then minimize mean up-Brier, "
            "then minimize mean tick-MAE"
        ),
        "evaluation_segment": "validation",
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
