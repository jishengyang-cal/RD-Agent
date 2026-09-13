"""
Model workflow with session control
"""

import asyncio
from typing import Any

import fire

from rdagent.app.lob_model_loop import run_lob_pool
from rdagent.app.qlib_rd_loop.conf import MODEL_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError


class ModelRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)


def main(
    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,  # noqa: FBT001, FBT002 -- preserve existing positional CLI/API
    base_features_path: str | None = None,
    lob_pool: str | None = None,
    qlib_python: str | None = None,
    research_root: str | None = None,
    lob_tracking_uri: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Auto R&D Evolving loop for fintech models

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/model.py $LOG_PATH/__session__/1/0_propose --step_n 1

    """
    if kwargs.get("help") is True or kwargs.get("h") is True:
        # Fire may otherwise consume --help as **kwargs and invoke the workflow.
        fire.Fire(main, command=["--", "--help"])
        return
    if lob_pool is not None:
        if any(value is not None for value in (path, step_n, loop_n, all_duration, base_features_path)):
            message = "LOB mode does not accept generic RD-loop/session arguments"
            raise ValueError(message)
        if qlib_python is None or research_root is None:
            message = "LOB mode requires qlib_python and research_root"
            raise ValueError(message)

        run_lob_pool(
            pool=lob_pool,
            qlib_python=qlib_python,
            research_root=research_root,
            **({"tracking_uri": lob_tracking_uri} if lob_tracking_uri is not None else {}),
        )
        return
    if qlib_python is not None or research_root is not None or lob_tracking_uri is not None:
        message = "qlib_python/research_root/lob_tracking_uri are only valid with a LOB pool"
        raise ValueError(message)
    model_loop = ModelRDLoop(MODEL_PROP_SETTING) if path is None else ModelRDLoop.load(path, checkout=checkout)
    model_loop._init_base_features(base_features_path)  # noqa: SLF001 -- existing RDLoop initialization protocol
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        model_loop._set_interactor(*kwargs["user_interaction_queues"])  # noqa: SLF001 -- existing RDLoop protocol
        model_loop._interact_init_params()  # noqa: SLF001 -- existing RDLoop protocol
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
