"""Bounded synthetic SDK experiment; not a production research scheduler."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import optuna
from mlflow.tracking import MlflowClient
from optuna.trial import TrialState

from rdagent.app.lob_model_loop import run_lob_pool, validate_lob_pool

if not __debug__:
    raise RuntimeError(
        "This integration test requires Python assertions; do not use -O"
    )

ROOT = Path(os.environ["LOB_OPTUNA_PILOT_ROOT"]).expanduser().resolve(strict=True)
# Hold one writer for the entire process; never take over another live pilot.
LOCK = (ROOT / ".driver.lock").open("a")
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
FROZEN = json.loads((ROOT / "preregistration.json").read_text())
if FROZEN["synthetic_only"] is not True or optuna.__version__ != "4.8.0":
    raise ValueError("Expected the frozen synthetic Optuna 4.8.0 pilot")
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != FROZEN["driver_sha256"]:
    raise ValueError("Driver changed after preparation")
if (
    hashlib.sha256((ROOT / "spec-7.json").read_bytes()).hexdigest()
    != FROZEN["base_spec_sha256"]
):
    raise ValueError("Base specification changed")
if (
    str(Path(os.environ["RESEARCH_ROOT"]).resolve(strict=True))
    != FROZEN["research_root"]
):
    raise ValueError("Research checkout changed")
if str(Path(os.environ["QLIB_PYTHON"]).absolute()) != FROZEN["qlib_python"]:
    raise ValueError("Training interpreter changed")
for key, path in [
    ("research_commit", FROZEN["research_root"]),
    ("rd_agent_commit", str(Path(__file__).resolve().parents[2])),
]:
    if (
        subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
        != FROZEN[key]
    ):
        raise ValueError("Source revision changed")
for package, version in FROZEN["versions"].items():
    if importlib.metadata.version(package) != version:
        raise ValueError("Dependency version changed")
STORAGE = f"sqlite:///{ROOT / 'optuna.db'}"
CONTRACT = hashlib.sha256((ROOT / "preregistration.json").read_bytes()).hexdigest()
BASE = json.loads((ROOT / "spec-7.json").read_text())
optuna.logging.set_verbosity(optuna.logging.WARNING)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_for(name):
    sampler = (
        optuna.samplers.RandomSampler(seed=20260913)
        if name == "random"
        else optuna.samplers.TPESampler(seed=20260913, n_startup_trials=1)
    )
    if stage == "first":
        study = optuna.create_study(
            study_name=f"lob-software-{name}",
            storage=STORAGE,
            direction="maximize",
            sampler=sampler,
            load_if_exists=False,
        )
    else:
        study = optuna.load_study(
            study_name=f"lob-software-{name}",
            storage=STORAGE,
            sampler=sampler,
        )
    existing = study.user_attrs.get("contract_sha256")
    if existing is None:
        if stage != "first" or study.trials:
            raise ValueError("Unbound existing study")
        study.set_user_attr("contract_sha256", CONTRACT)
    elif existing != CONTRACT:
        raise ValueError("Pilot contract changed")
    return study


def request(study, name):
    if len(study.trials) >= 2:
        raise ValueError("Fixed two-trial budget exhausted")
    trial = study.ask()
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
    spec = json.loads(json.dumps(BASE))
    spec["model"]["learning_rate"] = learning_rate
    spec["output_root"] = str(ROOT / "runs" / name)
    path = ROOT / f"{name}-{trial.number}-spec.json"
    with path.open("x") as stream:
        json.dump(spec, stream, indent=2, sort_keys=True)
    pool = ROOT / f"{name}-{trial.number}-pool.json"
    with pool.open("x") as stream:
        json.dump(
            {
                "schema_version": "lob-challenger-pool/v1",
                "pool_id": f"optuna-{name}-{trial.number}",
                "candidates": [str(path)],
                "top_k": 1,
            },
            stream,
        )
    validate_lob_pool(pool)
    trial.set_user_attr(
        "request",
        {
            "pool": str(pool),
            "pool_sha256": digest(pool),
            "spec": str(path),
            "spec_sha256": digest(path),
            "readiness_sha256": digest(spec["readiness_path"]),
        },
    )
    return trial.number


def verified_request(trial):
    bound = trial.user_attrs["request"]
    if (
        digest(bound["pool"]) != bound["pool_sha256"]
        or digest(bound["spec"]) != bound["spec_sha256"]
    ):
        raise ValueError("Requested candidate changed")
    _, candidates = validate_lob_pool(bound["pool"])
    spec = candidates[0][1]
    if digest(spec["readiness_path"]) != bound["readiness_sha256"]:
        raise ValueError("Readiness changed")
    return bound, spec


def verify_result(study, trial):
    bound, spec = verified_request(trial)
    receipt = study.user_attrs[f"result_{trial.number}"]
    if digest(receipt["registry"]) != receipt["registry_sha256"]:
        raise ValueError("Registry changed")
    result = json.loads(Path(receipt["registry"]).read_text())
    assert result["evaluation_segment"] == "validation"
    assert result["candidate_count"] == result["completed_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["source_spec_sha256"] == bound["spec_sha256"]
    folder = Path(spec["output_root"]) / candidate["run_id"]
    for field, filename in (
        ("checkpoint_sha256", "model.pt"),
        ("metrics_sha256", "metrics.json"),
        ("prediction_bundle_sha256", "prediction-bundle.json"),
    ):
        assert digest(folder / filename) == candidate[field]
    assert candidate["audit"]["artifact_valid"] is True
    assert candidate["audit"]["evaluation_segment"] == "validation"
    value = candidate["ranking"]["mean_direction_f1_macro"]
    if trial.state == TrialState.COMPLETE:
        assert trial.value == value
    tracking_db = Path(spec["output_root"]) / "mlflow.db"
    if (
        not tracking_db.is_file()
        or receipt["tracking_uri"] != f"sqlite:///{tracking_db}"
    ):
        raise ValueError("Expected the existing candidate tracking database")
    client = MlflowClient(tracking_uri=receipt["tracking_uri"])
    recorder = client.get_run(receipt["recorder_id"])
    assert recorder.info.status == "FINISHED"
    assert recorder.data.metrics["progress_optimizer_updates"] == 1.0
    assert recorder.data.params["spec_sha256"] == bound["spec_sha256"]
    assert recorder.data.tags["optuna.study_name"] == study.study_name
    assert recorder.data.tags["optuna.trial_number"] == str(trial.number)
    assert recorder.data.tags["optuna.contract_sha256"] == CONTRACT
    return {
        "trial": trial.number,
        "state": trial.state.name,
        "learning_rate": trial.params["learning_rate"],
        "value": value,
        "run_id": candidate["run_id"],
        "registry": receipt["registry"],
        "recorder_id": receipt["recorder_id"],
        "recorder_status": recorder.info.status,
        "baseline_screen": candidate["audit"]["baseline_screen"],
        "shortlist_count": len(result["shortlist"]),
    }


def execute(study, trial):
    bound, spec = verified_request(trial)
    if trial.state != TrialState.RUNNING:
        raise ValueError("Only a previously requested running trial may execute")
    if f"result_{trial.number}" in study.user_attrs:
        raise ValueError(
            "Result exists without final trial state; inspect before continuing"
        )
    result = run_lob_pool(
        pool=bound["pool"],
        qlib_python=os.environ["QLIB_PYTHON"],
        research_root=os.environ["RESEARCH_ROOT"],
        tracking_uri=f"sqlite:///{Path(spec['output_root']) / 'mlflow.db'}",
    )
    candidate = result["candidates"][0]
    context = json.loads(
        (
            Path(spec["output_root"])
            / "tracking"
            / candidate["run_id"]
            / "tracking-context.json"
        ).read_text()
    )["attempts"][-1]
    assert context["spec_sha256"] == bound["spec_sha256"]
    assert context["run_id"] == candidate["run_id"]
    client = MlflowClient(tracking_uri=context["tracking_uri"])
    recorder = client.get_run(context["recorder_id"])
    assert recorder.info.status == "FINISHED"
    assert recorder.data.metrics["progress_optimizer_updates"] == 1.0
    assert recorder.data.params["spec_sha256"] == bound["spec_sha256"]
    reference = ROOT / f"{study.study_name}-{trial.number}-reference.json"
    with reference.open("x") as stream:
        json.dump(
            {
                "schema_version": "lob-optuna-pilot-reference/v1",
                "synthetic_only": True,
                "study_name": study.study_name,
                "trial_number": trial.number,
                "params": trial.params,
                "contract_sha256": CONTRACT,
                "spec_sha256": bound["spec_sha256"],
                "run_id": candidate["run_id"],
                "registry_sha256": digest(result["registry"]),
                "objective": "validation mean_direction_f1_macro",
            },
            stream,
            sort_keys=True,
        )
    client.log_artifact(
        context["recorder_id"], str(reference), artifact_path="optuna-trials"
    )
    downloaded = client.download_artifacts(
        context["recorder_id"], f"optuna-trials/{reference.name}"
    )
    assert digest(downloaded) == digest(reference)
    for key, value in {
        "optuna.study_name": study.study_name,
        "optuna.trial_number": str(trial.number),
        "optuna.contract_sha256": CONTRACT,
    }.items():
        client.set_tag(context["recorder_id"], key, value)
    study.set_user_attr(
        f"result_{trial.number}",
        {
            "registry": result["registry"],
            "registry_sha256": digest(result["registry"]),
            "tracking_uri": context["tracking_uri"],
            "recorder_id": context["recorder_id"],
        },
    )
    # This score never consumes the held-out test segment or changes the audit gate.
    study.tell(trial.number, candidate["ranking"]["mean_direction_f1_macro"])


parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=("first", "resume", "verify"))
stage = parser.parse_args().stage
if stage != "first" and not (ROOT / "optuna.db").is_file():
    raise ValueError("Existing Optuna database required; recovery never creates one")
if (ROOT / f"{stage}-result.json").exists():
    raise ValueError(
        "Stage receipt already exists; preserve it and inspect the completed result"
    )
summaries = {}
for name in ("random", "tpe"):
    study = study_for(name)
    if stage == "first":
        if study.trials:
            raise ValueError("First stage requires an empty study")
        number = request(study, name)
        execute(study, study.get_trials()[number])
        request(
            study, name
        )  # Persist the next request, then leave execution for a new process.
    elif stage == "resume":
        if len(study.trials) != 2:
            raise ValueError("Expected the fixed requested trial set")
        for trial in study.get_trials():
            if trial.state == TrialState.RUNNING:
                execute(study, trial)
            elif trial.state != TrialState.COMPLETE:
                raise ValueError("Unexpected trial state")
    summaries[name] = [
        (
            verify_result(study, trial)
            if trial.state == TrialState.COMPLETE
            else {"trial": trial.number, "state": trial.state.name}
        )
        for trial in study.get_trials()
    ]
    if stage == "verify":
        assert len(study.trials) == 2 and all(
            t.state == TrialState.COMPLETE for t in study.trials
        )
report = {
    "synthetic_only": True,
    "stage": stage,
    "contract_sha256": CONTRACT,
    "optuna_version": optuna.__version__,
    "studies": summaries,
    "performance_or_market_advantage_verified": False,
}
with (ROOT / f"{stage}-result.json").open("x") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
print(json.dumps(report, indent=2))
