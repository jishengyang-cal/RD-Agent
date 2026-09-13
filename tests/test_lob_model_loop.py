import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rdagent.app import lob_model_loop
from rdagent.app.lob_model_loop import (
    HORIZONS_MS,
    _audit_candidate,
    _load_candidate_result,
    _run_id,
    _runner,
    _sha256,
    run_lob_pool,
    validate_lob_pool,
    validate_lob_spec,
)


def _spec(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": "strict-l2-training-readiness/v1",
                "formal_ready": True,
                "strict_l2_only": True,
            },
        ),
    )
    value = {
        "schema_version": "lob-experiment-spec/v1",
        "readiness_path": str(readiness),
        "window": {"name": "roll-00", "sealed_final": False},
        "architecture": "wide_tlob",
        "variant": "shared",
        "segments": {"train": {}, "valid": {}, "test": {}},
        "seed": 7,
        "dataset": {"context": 128, "medium_context": 300, "history_days": 5, "stride": 10},
        "model": {
            "kwargs": {},
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 4,
            "device": "cpu",
            "early_stop": 1,
            "gradient_clip": 3.0,
            "num_workers": 0,
            "prefetch_factor": 2,
            "direction_loss_weight": 0.25,
        },
        "output_root": str(tmp_path / "runs"),
        "strict_l2_only": True,
        "use_historical_vap": False,
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


def test_pool_observation_keeps_five_candidates_attempts_and_missing_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, spec = _spec(tmp_path)
    paths = []
    for architecture in sorted(lob_model_loop.ALLOWED_ARCHITECTURES):
        path = tmp_path / f"{architecture}.json"
        value = {**spec, "architecture": architecture}
        path.write_text(json.dumps(value))
        paths.append(str(path))
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": lob_model_loop.LOB_POOL_SCHEMA,
                "pool_id": "observation",
                "candidates": paths,
                "top_k": 1,
            }
        )
    )
    queries = []

    class Page(list):
        token = None

    class Client:
        def search_runs(self, **kwargs: Any) -> Page:
            queries.append(kwargs)
            path = Path(paths[0])
            value = json.loads(path.read_text())
            if _run_id(value) not in kwargs["filter_string"]:
                return Page()
            resumed = kwargs["page_token"] is not None
            run = SimpleNamespace(
                info=SimpleNamespace(
                    run_id="attempt-2" if resumed else "attempt-1",
                    experiment_id="1",
                    status="FINISHED" if resumed else "FAILED",
                    start_time=2 if resumed else 1,
                    end_time=3,
                ),
                data=SimpleNamespace(
                    params={
                        "run_id": _run_id(value),
                        "spec_sha256": "wrong" if resumed else _sha256(path),
                        "architecture": value["architecture"],
                        "sealed_final": "False",
                    },
                    metrics={"epoch_train_loss": 1.0},
                    tags={"audit.screen_status": "rejected"},
                ),
            )
            page = Page([run])
            page.token = None if resumed else "next"
            return page

    def forbid(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("observation must not launch training or audit subprocesses")

    monkeypatch.setattr(lob_model_loop.subprocess, "run", forbid)
    result = lob_model_loop.inspect_lob_pool(pool=str(pool), tracking_client=Client(), experiment_ids=["1"])
    assert result["candidate_count"] == len(paths)
    assert not result["ranking_performed"]
    assert not result["process_liveness_verified"]
    assert len(queries) == len(paths) + 1
    assert all(row["tracking_state"] == "unknown" for row in result["candidates"][1:])
    attempts = result["candidates"][0]["attempts"]
    assert [row["execution_status"] for row in attempts] == ["FAILED", "FINISHED"]
    assert attempts[0]["identity_valid"]
    assert not attempts[1]["identity_valid"]
    assert attempts[1]["recorded_metrics"] == attempts[1]["recorded_summary"] == {}
    assert not Path(spec["output_root"]).exists()


def test_pool_observation_real_mlflow_roundtrip(tmp_path: Path) -> None:
    """Software-only records: no market training or historical-result import."""
    tracking = pytest.importorskip("mlflow.tracking")

    _, spec = _spec(tmp_path)
    client = tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment_id = client.create_experiment(
        "synthetic-pool-observation", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    paths = []
    recorded_count = 3
    for index, architecture in enumerate(sorted(lob_model_loop.ALLOWED_ARCHITECTURES)):
        path = tmp_path / f"{architecture}.json"
        value = {**spec, "architecture": architecture}
        path.write_text(json.dumps(value))
        paths.append(str(path))
        if index >= recorded_count:
            continue
        run_id = client.create_run(experiment_id).info.run_id
        for key, parameter in {
            "run_id": _run_id(value),
            "spec_sha256": _sha256(path),
            "sealed_final": "False",
            "architecture": architecture,
        }.items():
            client.log_param(run_id, key, parameter)
        if index != 1:
            client.set_terminated(run_id, "FINISHED" if index == 0 else "FAILED")
        if index == 0:
            client.set_tag(run_id, "audit.screen_status", "rejected")
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": lob_model_loop.LOB_POOL_SCHEMA,
                "pool_id": "sdk-observation",
                "candidates": paths,
                "top_k": 1,
            }
        )
    )
    result = lob_model_loop.inspect_lob_pool(pool=str(pool), tracking_client=client, experiment_ids=[experiment_id])
    assert result["candidate_count"] == len(paths)
    assert [row["tracking_state"] for row in result["candidates"]] == [
        "observed",
        "observed",
        "observed",
        "unknown",
        "unknown",
    ]
    assert [row["attempts"][0]["execution_status"] for row in result["candidates"][:3]] == [
        "FINISHED",
        "RUNNING",
        "FAILED",
    ]
    assert result["candidates"][0]["attempts"][0]["recorded_summary"]["audit.screen_status"] == "rejected"
    assert all(row["attempts"][0]["identity_valid"] for row in result["candidates"][:3])
    assert len(client.search_runs([experiment_id])) == recorded_count


@pytest.mark.parametrize("wrong_binding", [False, True])
def test_audit_forwards_only_explicit_matching_recorder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_binding: bool,
) -> None:
    spec_path, spec = _spec(tmp_path)
    run_root = Path(spec["output_root"]) / _run_id(spec)
    context = run_root.parent / "tracking" / run_root.name / "tracking-context.json"
    context.parent.mkdir(parents=True)
    uri = "sqlite:///synthetic-existing.db"
    context.write_text(
        json.dumps(
            {
                "schema_version": "lob-tracking-context/v1",
                "attempts": [
                    {
                        "tracking_uri": uri,
                        "run_id": run_root.name,
                        "recorder_id": "synthetic-recorder",
                        "spec_sha256": "wrong" if wrong_binding else _sha256(spec_path),
                    }
                ],
            }
        )
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "audit_lob_candidate.py").write_text(
        "import json,sys\nprint(json.dumps({'artifact_valid': True, "
        "'evaluation_segment':'validation', 'args':sys.argv[1:]}))\n"
    )
    if wrong_binding:

        def forbid(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("mismatched context must not launch the auditor")

        monkeypatch.setattr(lob_model_loop.subprocess, "run", forbid)
        with pytest.raises(ValueError, match="identity mismatch"):
            _audit_candidate(Path(sys.executable), tmp_path, run_root, spec_path, tracking_uri=uri)
    else:
        result = _audit_candidate(Path(sys.executable), tmp_path, run_root, spec_path, tracking_uri=uri)
        assert result["args"][-4:] == ["--tracking-uri", uri, "--recorder-id", "synthetic-recorder"]


def test_model_loop_forwards_tracking_only_for_lob_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Import the actual CLI module with unrelated LLM provider dependencies isolated.
    source = Path(lob_model_loop.__file__).parent / "qlib_rd_loop" / "model.py"
    monkeypatch.setitem(sys.modules, "rdagent.app.qlib_rd_loop.conf", SimpleNamespace(MODEL_PROP_SETTING=object()))
    monkeypatch.setitem(sys.modules, "rdagent.components.workflow.rd_loop", SimpleNamespace(RDLoop=object))
    monkeypatch.setitem(sys.modules, "rdagent.core.exception", SimpleNamespace(ModelEmptyError=RuntimeError))
    calls = []
    monkeypatch.setattr(lob_model_loop, "run_lob_pool", lambda **kwargs: calls.append(kwargs))
    module_spec = importlib.util.spec_from_file_location("lob_cli_under_test", source)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.main(
        lob_pool="pool.json", qlib_python="python", research_root="research", lob_tracking_uri="sqlite:///existing.db"
    )
    assert calls == [
        {
            "pool": "pool.json",
            "qlib_python": "python",
            "research_root": "research",
            "tracking_uri": "sqlite:///existing.db",
        }
    ]
    with pytest.raises(ValueError, match="only valid with a LOB pool"):
        module.main(lob_tracking_uri="sqlite:///existing.db")
    for flag in ("help", "h"):
        with pytest.raises(SystemExit) as stopped:
            module.main(**{flag: True})
        assert stopped.value.code == 0
    assert len(calls) == 1  # Neither help form may create or launch another loop.


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
    metrics = {"overall": {}, "evaluation_segment": "validation"}
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
        implementation,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    implementation["sha256"] = hashlib.sha256(payload).hexdigest()
    (run_root / "implementation.json").write_text(json.dumps(implementation))


def test_lob_pool_ranks_completed_immutable_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "round-01",
                "candidates": [str(first), str(second)],
                "top_k": 1,
            },
        ),
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("# test runner\n")
    expected_candidates = 2
    monkeypatch.setattr(
        lob_model_loop,
        "_audit_candidate",
        lambda *_args: {
            "artifact_valid": True,
            "evaluation_segment": "validation",
            "baseline_screen": {"screening_effective": True},
        },
    )
    assert len(validate_lob_pool(pool)[1]) == expected_candidates
    result = run_lob_pool(
        pool=str(pool),
        qlib_python=str(Path("/bin/true")),
        research_root=str(tmp_path),
    )
    registry = json.loads(Path(result["registry"]).read_text())
    assert registry["shortlist"][0]["architecture"] == "mlp"
    assert registry["completed_count"] == expected_candidates


def test_lob_pool_keeps_audited_rejection_out_of_shortlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _spec(tmp_path)
    _publish_candidate(candidate, f1=0.6)
    pool = tmp_path / "rejected-pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "rejected-round",
                "candidates": [str(candidate)],
                "top_k": 1,
            },
        ),
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("# test runner\n")
    monkeypatch.setattr(
        lob_model_loop,
        "_audit_candidate",
        lambda *_args: {
            "artifact_valid": True,
            "evaluation_segment": "validation",
            "baseline_screen": {"screening_effective": False},
        },
    )
    result = run_lob_pool(
        pool=str(pool),
        qlib_python="/bin/true",
        research_root=str(tmp_path),
    )
    assert result["completed_count"] == 1
    assert result["shortlist"] == []
    assert result["candidates"][0]["audit"]["artifact_valid"] is True


def test_lob_pool_does_not_rank_partial_candidate_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_value = _spec(tmp_path)
    second = tmp_path / "candidate-2.json"
    second_value = dict(first_value)
    second_value["architecture"] = "mlp"
    second_value["variant"] = "baseline"
    second.write_text(json.dumps(second_value))
    _publish_candidate(first, f1=0.4)
    pool = tmp_path / "partial-pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "partial-round",
                "candidates": [str(first), str(second)],
                "top_k": 1,
            },
        ),
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("# test runner\n")
    monkeypatch.setattr(
        lob_model_loop,
        "_audit_candidate",
        lambda *_args: {
            "artifact_valid": True,
            "evaluation_segment": "validation",
            "baseline_screen": {"screening_effective": True},
        },
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        run_lob_pool(
            pool=str(pool),
            qlib_python="/bin/false",
            research_root=str(tmp_path),
        )
    assert not (Path(first_value["output_root"]) / "challenger-pool-partial-round.json").exists()


def test_lob_pool_rejects_noncomparable_split(tmp_path: Path) -> None:
    first, first_value = _spec(tmp_path)
    second = tmp_path / "candidate-2.json"
    second_value = dict(first_value)
    second_value["segments"] = {"train": {"end_ns": 1}, "valid": {}, "test": {}}
    second.write_text(json.dumps(second_value))
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "round-02",
                "candidates": [str(first), str(second)],
                "top_k": 1,
            },
        ),
    )
    with pytest.raises(ValueError, match="same evaluation cell"):
        validate_lob_pool(pool)


@pytest.mark.parametrize("copied", [False, True])
def test_lob_pool_rejects_duplicate_identity_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    copied: bool,
) -> None:
    first, value = _spec(tmp_path)
    second = tmp_path / "copied-spec.json" if copied else first
    second.write_text(json.dumps(value, indent=2, sort_keys=True))
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "duplicate-round",
                "candidates": [str(first), str(second)],
                "top_k": 2,
            },
        ),
    )
    launches = []
    monkeypatch.setattr(lob_model_loop.subprocess, "run", lambda *args, **_kwargs: launches.append(args))
    with pytest.raises(ValueError, match="duplicate candidate run ID"):
        run_lob_pool(pool=str(pool), qlib_python=sys.executable, research_root=str(tmp_path))
    assert launches == []
    assert not Path(value["output_root"]).exists()


def test_lob_pool_audits_with_expanded_research_root(tmp_path: Path) -> None:
    candidate, _ = _spec(tmp_path)
    _publish_candidate(candidate, f1=0.6)
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "expanded-root",
                "candidates": [str(candidate)],
                "top_k": 1,
            },
        ),
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("")
    (scripts / "audit_lob_candidate.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'artifact_valid': True, 'evaluation_segment': 'validation', 'cwd': os.getcwd(), "
        "'args': sys.argv[1:], 'baseline_screen': {'screening_effective': True}}))\n",
    )
    result = run_lob_pool(
        pool=str(pool),
        qlib_python=sys.executable,
        research_root="~/" + os.path.relpath(tmp_path, Path.home()),
    )
    registry = json.loads(Path(result["registry"]).read_text())
    audit = registry["shortlist"][0]["audit"]
    assert audit["cwd"] == str(tmp_path.resolve())
    assert audit["args"] == [
        str(tmp_path / "runs" / registry["shortlist"][0]["run_id"]),
        "--source-spec",
        str(candidate),
    ]


@pytest.mark.parametrize("segment", [None, "test", "legacy_test", "development_test", "sealed_final"])
def test_ranking_rejects_every_non_validation_segment(tmp_path: Path, segment: str | None) -> None:
    path, spec = _spec(tmp_path)
    _publish_candidate(path, f1=0.99)
    metrics_path = Path(spec["output_root"]) / _run_id(spec) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["evaluation_segment"] = segment
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="validation"):
        _load_candidate_result(spec, path)


@pytest.mark.parametrize("segment", [None, "development_test", "sealed_final"])
def test_pool_requires_validation_audit_not_just_validation_metrics(
    tmp_path: Path,
    segment: str | None,
) -> None:
    candidate, spec = _spec(tmp_path)
    _publish_candidate(candidate, f1=0.99)
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": "audit-segment",
                "candidates": [str(candidate)],
                "top_k": 1,
            },
        ),
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_lob_experiment.py").write_text("")
    audit = {
        "artifact_valid": True,
        "evaluation_segment": segment,
        "baseline_screen": {"screening_effective": True},
    }
    (scripts / "audit_lob_candidate.py").write_text(
        f"print({json.dumps(audit)!r})\n",
    )
    with pytest.raises(RuntimeError, match="failed"):
        run_lob_pool(pool=str(pool), qlib_python=sys.executable, research_root=str(tmp_path))
    assert not (Path(spec["output_root"]) / "challenger-pool-audit-segment.json").exists()
