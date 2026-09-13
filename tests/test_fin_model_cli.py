import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rdagent.app import lob_model_loop


@pytest.fixture
def public_cli(monkeypatch):
    calls = []

    def forbid(*args, **kwargs):
        pytest.fail("Unrelated command must not run")

    dependencies = {
        "rdagent.app.data_science.loop": "main",
        "rdagent.app.finetune.llm.loop": "main",
        "rdagent.app.general_model.general_model": "extract_models_and_implement",
        "rdagent.app.qlib_rd_loop.factor": "main",
        "rdagent.app.qlib_rd_loop.factor_from_report": "main",
        "rdagent.app.qlib_rd_loop.quant": "main",
        "rdagent.app.utils.health_check": "health_check",
        "rdagent.app.utils.info": "collect_info",
        "rdagent.log.mle_summary": "grade_summary",
    }
    for module, attribute in dependencies.items():
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{attribute: forbid}))
    monkeypatch.setitem(
        sys.modules,
        "rdagent.app.qlib_rd_loop.model",
        SimpleNamespace(main=lambda **kwargs: calls.append(kwargs)),
    )
    spec = importlib.util.spec_from_file_location(
        "public_cli_under_test", Path(lob_model_loop.__file__).with_name("cli.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app, calls


def test_public_fin_model_help_exposes_tracking_without_launch(public_cli):
    app, calls = public_cli
    result = CliRunner().invoke(app, ["fin_model", "--help"], color=False)
    assert result.exit_code == 0, result.output
    assert "--lob-tracking-uri" in result.output
    assert calls == []


@pytest.mark.parametrize("tracking_uri", [None, "sqlite:///existing.db"])
def test_public_fin_model_forwards_lob_options(public_cli, tracking_uri):
    app, calls = public_cli
    args = ["fin_model", "--lob-pool", "pool.json", "--qlib-python", "python", "--research-root", "research"]
    if tracking_uri is not None:
        args.extend(["--lob-tracking-uri", tracking_uri])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == [{
        "path": None, "step_n": None, "loop_n": None, "all_duration": None,
        "checkout": True, "lob_pool": "pool.json", "qlib_python": "python",
        "research_root": "research", "lob_tracking_uri": tracking_uri,
    }]


def test_public_fin_model_preserves_session_invocation(public_cli):
    app, calls = public_cli
    result = CliRunner().invoke(app, [
        "fin_model", "--path", "session", "--step-n", "2", "--loop-n", "3",
        "--all-duration", "1h", "--no-checkout",
    ])
    assert result.exit_code == 0, result.output
    assert calls == [{
        "path": "session", "step_n": 2, "loop_n": 3, "all_duration": "1h",
        "checkout": False, "lob_pool": None, "qlib_python": None,
        "research_root": None, "lob_tracking_uri": None,
    }]
