"""Create small synthetic inputs for run_lob_optuna_pilot.py; never market evidence.

Requires the external Research checkout's existing manifest_fixture. Nothing is
registered as real market data. LOB_OPTUNA_PILOT_ROOT must name a new directory.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from research.lob.tensor_publication import READINESS_SCHEMA, UNIT_CONTRACT

root = Path(os.environ["LOB_OPTUNA_PILOT_ROOT"]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=False)
root.chmod(0o700)
research_root = Path(os.environ["RESEARCH_ROOT"]).resolve(strict=True)
if importlib.metadata.version("optuna") != "4.8.0":
    raise RuntimeError("This software pilot was specified for Optuna 4.8.0")
spec = importlib.util.spec_from_file_location(
    "fixture", research_root / "tests/unit/lob/test_qlib_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
publications = []
dates = ["2026-05-04", "2026-05-05", "2026-05-06"]
for date in dates:
    day = root / "data" / date
    day.mkdir(parents=True)
    tensor, source, times = module.manifest_fixture(day)
    raw = json.loads(source.read_text())
    raw["point_in_time"] = dict(known_at=date, effective_at=date)
    source.write_text(json.dumps(raw))
    manifest = json.loads(tensor.read_text())
    manifest["point_in_time"] = dict(known_at=date, effective_at=date)
    manifest["source_digest"] = module.digest(source)
    for entry in manifest["files"]:
        path = day / entry["path"]
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if "ts_recv" in arrays:
            arrays["ts_recv"] += pd.Timestamp(
                date + " 09:30:00", tz="America/New_York"
            ).value
        if entry["role"] == "future_path_labels_fast":
            for name in arrays:
                if name.startswith("delta_mid_ticks_"):
                    arrays[name] = np.resize(np.array([-1.0, 0.0, 1.0]), len(times))
        np.savez_compressed(path, **arrays)
        entry.update(size_bytes=path.stat().st_size, sha256=module.digest(path))
    tensor.write_text(json.dumps(manifest))
    publications.append(
        dict(
            tensor_manifest=str(tensor),
            tensor_manifest_sha256=module.digest(tensor),
            source_manifest=str(source),
            source_manifest_sha256=module.digest(source),
            files_verified=True,
        )
    )
readiness = root / "readiness.json"
readiness.write_text(
    json.dumps(
        dict(
            schema_version=READINESS_SCHEMA,
            authorization_mode="verified",
            formal_ready=True,
            strict_l2_only=True,
            use_historical_vap=False,
            unit_contract=UNIT_CONTRACT,
            expected_days=3,
            available_days=3,
            publications=publications,
            evidence_label="synthetic software fixture, not market readiness",
        )
    )
)
path = root / "spec-7.json"
path.write_text(
    json.dumps(
        dict(
            schema_version="lob-experiment-spec/v1",
            readiness_path=str(readiness),
            window=dict(name="synthetic-live", sealed_final=False),
            architecture="mlp",
            variant="shared",
            segments={
                key: dict(dates=[date], symbols=["TEST"])
                for key, date in zip(["train", "valid", "test"], dates)
            },
            seed=7,
            dataset=dict(context=4, medium_context=1, history_days=1, stride=20),
            model=dict(
                kwargs=dict(hidden=8),
                learning_rate=0.001,
                epochs=1,
                batch_size=4,
                device="cpu",
                early_stop=1,
                gradient_clip=3.0,
                num_workers=0,
                prefetch_factor=2,
                direction_loss_weight=0.25,
            ),
            output_root=str(root / "runs"),
            strict_l2_only=True,
            use_historical_vap=False,
            champion_lock=None,
        )
    )
)
print("Prepared synthetic-only inputs")

contract = {
    "synthetic_only": True,
    "research_commit": subprocess.check_output(
        ["git", "-C", str(research_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "rd_agent_commit": subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "research_root": str(research_root),
    "qlib_python": str(Path(os.environ["QLIB_PYTHON"]).absolute()),
    "versions": {
        name: importlib.metadata.version(name)
        for name in ["optuna", "mlflow", "pyqlib", "torch"]
    },
    "base_spec_sha256": hashlib.sha256((root / "spec-7.json").read_bytes()).hexdigest(),
    "driver_sha256": hashlib.sha256(
        Path(__file__).with_name("run_lob_optuna_pilot.py").read_bytes()
    ).hexdigest(),
    "budget": "two trials each for RandomSampler and TPE; one CPU update per candidate",
    "objective": "validation mean_direction_f1_macro; independent baseline gate unchanged",
    "limitations": "software fixture only; not performance, market efficacy, pruning, or a production scheduler",
}
(root / "preregistration.json").write_text(
    json.dumps(contract, indent=2, sort_keys=True)
)
