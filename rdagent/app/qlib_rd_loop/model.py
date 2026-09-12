"""
Model workflow with session control
"""

import asyncio

import fire
from rdagent.app.qlib_rd_loop.conf import MODEL_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError


class ModelRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    base_features_path: str | None = None,
    lob_spec: str | None = None,
    lob_pool: str | None = None,
    qlib_python: str | None = None,
    research_root: str | None = None,
    **kwargs,
):
    """
    Auto R&D Evolving loop for fintech models

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/model.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if lob_spec is not None or lob_pool is not None:
        if lob_spec is not None and lob_pool is not None:
            raise ValueError("LOB mode accepts either lob_spec or lob_pool, not both")
        if any(value is not None for value in (path, step_n, loop_n, all_duration, base_features_path)):
            raise ValueError("LOB mode does not accept generic RD-loop/session arguments")
        if qlib_python is None or research_root is None:
            raise ValueError("LOB mode requires qlib_python and research_root")
        from rdagent.app.lob_model_loop import run_lob_pool, run_lob_spec

        if lob_pool is not None:
            run_lob_pool(pool=lob_pool, qlib_python=qlib_python, research_root=research_root)
        else:
            run_lob_spec(spec=lob_spec, qlib_python=qlib_python, research_root=research_root)
        return
    if qlib_python is not None or research_root is not None:
        raise ValueError("qlib_python/research_root are only valid with a LOB spec or pool")
    if path is None:
        model_loop = ModelRDLoop(MODEL_PROP_SETTING)
    else:
        model_loop = ModelRDLoop.load(path, checkout=checkout)
    model_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        model_loop._set_interactor(*kwargs["user_interaction_queues"])
        model_loop._interact_init_params()
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
