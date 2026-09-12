import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from rdagent.app import lob_model_loop
from rdagent.app.lob_model_loop import (
    HORIZONS_MS,
    _run_id,
    _runner,
    run_lob_pool,
    validate_lob_pool,
    validate_lob_spec,
)
from rdagent.app.lob_pool_control import create_recovery_pool, inspect_lob_pool


def _spec(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({
        "schema_version": "strict-l2-training-readiness/v1",
        "formal_ready": True,
        "strict_l2_only": True,
    }))
    value = {
        "schema_version": "lob-experiment-spec/v1",
        "readiness_path": str(readiness),
        "window": {"name": "roll-00", "sealed_final": False},
        "architecture": "wide_tlob", "variant": "shared",
        "segments": {"train": {}, "valid": {}, "test": {}}, "seed": 7,
        "dataset": {"context": 128, "medium_context": 300, "history_days": 5, "stride": 10},
        "model": {
            "kwargs": {}, "learning_rate": 0.001, "epochs": 1,
            "batch_size": 4, "device": "cpu", "early_stop": 1,
            "gradient_clip": 3.0, "num_workers": 0,
            "prefetch_factor": 2, "direction_loss_weight": 0.25,
        },
        "output_root": str(tmp_path / "runs"),
        "strict_l2_only": True, "use_historical_vap": False,
        "champion_lock": None,
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value))
    return path, value


def test_lob_spec_accepts_only_model_training_surface(tmp_path: Path) -> None:
    path, value = _spec(tmp_path)
    assert validate_lob_spec(path)["architecture"] == "wide_tlob"
    value["factor"] = "WallScore"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="forbidden"):
        validate_lob_spec(path)


def test_lob_spec_rejects_sealed_final_and_non_l2(tmp_path: Path) -> None:
    path, value = _spec(tmp_path)
    value["window"]["sealed_final"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="sealed"):
        validate_lob_spec(path)
    value["window"]["sealed_final"] = False
    value["strict_l2_only"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="strict-L2"):
        validate_lob_spec(path)


def test_runner_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/bin/true")
    scripts = tmp_path / "research/scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_lob_experiment.py").write_text("# fixture\n")
    selected_python, _ = _runner(str(python), str(tmp_path / "research"))
    assert selected_python == python.absolute()
    assert selected_python.is_symlink()


def _publish_candidate(spec_path: Path, *, f1: float) -> None:
    spec = json.loads(spec_path.read_text())
    run_root = Path(spec["output_root"]) / _run_id(spec)
    run_root.mkdir(parents=True)
    metrics = {"overall": {}}
    for horizon in HORIZONS_MS:
        metrics["overall"][f"direction_f1_macro_{horizon}ms"] = f1
        metrics["overall"][f"up_brier_{horizon}ms"] = 0.2
        metrics["overall"][f"mae_{horizon}ms"] = 1.0
    (run_root / "metrics.json").write_text(json.dumps(metrics))
    (run_root / "experiment-spec.json").write_text(json.dumps(spec))
    (run_root / "prediction-bundle.json").write_text("{}")
    (run_root / "model.pt").write_bytes(b"checkpoint")
    implementation = {
        "schema_version": "lob-implementation/v1",
        "python": "3.12.0",
        "packages": {},
        "files": {"models.py": "0" * 64},
    }
    payload = json.dumps(
        implementation, sort_keys=True, separators=(",", ":"),
    ).encode()
    implementation["sha256"] = hashlib.sha256(payload).hexdigest()
    (run_root / "implementation.json").write_text(json.dumps(implementation))


def test_lob_pool_ranks_completed_immutable_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_value = _spec(tmp_path)
    second = tmp_path / "candidate-2.json"
    second_value = dict(first_value)
    second_value["architecture"] = "mlp"
    second_value["variant"] = "baseline"
    second.write_text(json.dumps(second_value))
    _publish_candidate(first, f1=0.4)
    _publish_candidate(second, f1=0.6)
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "schema_version": "lob-challenger-pool/v1",
        "pool_id": "round-01",
        "candidates": [str(first), str(second)],
        "top_k": 1,
    }))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("# test runner\n")
    expected_candidates = 2
    monkeypatch.setattr(
        lob_model_loop, "_audit_candidate",
        lambda *_args: {
            "artifact_valid": True,
            "baseline_screen": {"screening_effective": True},
        },
    )
    assert len(validate_lob_pool(pool)[1]) == expected_candidates
    result = run_lob_pool(
        pool=str(pool), qlib_python=str(Path("/bin/true")),
        research_root=str(tmp_path),
    )
    registry = json.loads(Path(result["registry"]).read_text())
    assert registry["shortlist"][0]["architecture"] == "mlp"
    assert registry["completed_count"] == expected_candidates


def test_lob_pool_keeps_audited_rejection_out_of_shortlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _spec(tmp_path)
    _publish_candidate(candidate, f1=0.6)
    pool = tmp_path / "rejected-pool.json"
    pool.write_text(json.dumps({
        "schema_version": "lob-challenger-pool/v1",
        "pool_id": "rejected-round",
        "candidates": [str(candidate)],
        "top_k": 1,
    }))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("# test runner\n")
    monkeypatch.setattr(
        lob_model_loop, "_audit_candidate",
        lambda *_args: {
            "artifact_valid": True,
            "baseline_screen": {"screening_effective": False},
        },
    )
    result = run_lob_pool(
        pool=str(pool), qlib_python="/bin/true", research_root=str(tmp_path),
    )
    assert result["completed_count"] == 1
    assert result["shortlist"] == []
    assert result["candidates"][0]["audit"]["artifact_valid"] is True


def test_lob_pool_does_not_rank_partial_candidate_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_value = _spec(tmp_path)
    second = tmp_path / "candidate-2.json"
    second_value = dict(first_value)
    second_value["architecture"] = "mlp"
    second_value["variant"] = "baseline"
    second.write_text(json.dumps(second_value))
    _publish_candidate(first, f1=0.4)
    pool = tmp_path / "partial-pool.json"
    pool.write_text(json.dumps({
        "schema_version": "lob-challenger-pool/v1",
        "pool_id": "partial-round",
        "candidates": [str(first), str(second)],
        "top_k": 1,
    }))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("# test runner\n")
    monkeypatch.setattr(
        lob_model_loop, "_audit_candidate",
        lambda *_args: {
            "artifact_valid": True,
            "baseline_screen": {"screening_effective": True},
        },
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        run_lob_pool(
            pool=str(pool), qlib_python="/bin/false", research_root=str(tmp_path),
        )
    assert not (Path(first_value["output_root"]) / "challenger-pool-partial-round.json").exists()


def test_lob_pool_rejects_noncomparable_split(tmp_path: Path) -> None:
    first, first_value = _spec(tmp_path)
    second = tmp_path / "candidate-2.json"
    second_value = dict(first_value)
    second_value["segments"] = {"train": {"end_ns": 1}, "valid": {}, "test": {}}
    second.write_text(json.dumps(second_value))
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "schema_version": "lob-challenger-pool/v1",
        "pool_id": "round-02",
        "candidates": [str(first), str(second)],
        "top_k": 1,
    }))
    with pytest.raises(ValueError, match="same evaluation cell"):
        validate_lob_pool(pool)


def test_pool_status_distinguishes_complete_incomplete_and_pending(tmp_path: Path) -> None:
    first, first_value = _spec(tmp_path)
    second = tmp_path / "candidate-2.json"
    second_value = dict(first_value)
    second_value["architecture"] = "mlp"
    second_value["variant"] = "baseline"
    second.write_text(json.dumps(second_value))
    third = tmp_path / "candidate-3.json"
    third_value = dict(first_value)
    third_value["architecture"] = "lstm"
    third_value["variant"] = "baseline"
    third.write_text(json.dumps(third_value))
    _publish_candidate(first, f1=0.4)
    incomplete = Path(first_value["output_root"]) / f".incomplete-{_run_id(second_value)}"
    incomplete.mkdir(parents=True)
    pool = tmp_path / "status-pool.json"
    pool.write_text(json.dumps({
        "schema_version": "lob-challenger-pool/v1",
        "pool_id": "status-round",
        "candidates": [str(first), str(second), str(third)],
        "top_k": 1,
    }))
    status = inspect_lob_pool(pool)
    assert status["counts"] == {"complete": 1, "incomplete": 1, "pending": 1, "invalid": 0}
    assert [candidate["state"] for candidate in status["candidates"]] == [
        "complete",
        "incomplete",
        "pending",
    ]


def test_recovery_pool_keeps_full_comparison_cell_and_never_overwrites(tmp_path: Path) -> None:
    first, _ = _spec(tmp_path)
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "schema_version": "lob-challenger-pool/v1",
        "pool_id": "failed-round",
        "candidates": [str(first)],
        "top_k": 1,
    }))
    recovery = tmp_path / "recovery.json"
    result = create_recovery_pool(pool, recovery, pool_id="failed-round-recovery")
    definition, candidates = validate_lob_pool(recovery)
    assert result["status"] == "complete"
    assert definition["pool_id"] == "failed-round-recovery"
    assert len(candidates) == 1
    with pytest.raises(FileExistsError, match="never overwrites"):
        create_recovery_pool(pool, recovery, pool_id="another-recovery")
