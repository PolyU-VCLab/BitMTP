"""Check scoring imports before model setup, without downloading datasets."""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os

TASK_DEPENDENCIES = {
    "gsm8k": {"lm_eval": ("simple_evaluate",), "datasets": ("load_dataset",)},
    "humaneval_plus": {
        "evalplus.data": ("get_human_eval_plus", "get_human_eval_plus_hash"),
    },
}


def preflight_task_dependencies(task):
    if task not in TASK_DEPENDENCIES:
        raise ValueError("published benchmarks support GSM8K and HumanEval+ only")
    if task == "humaneval_plus":
        if "HUMANEVAL_OVERRIDE_PATH" in os.environ:
            raise RuntimeError(
                "HumanEval+ requires the standard EvalPlus dataset; unset "
                "HUMANEVAL_OVERRIDE_PATH before evaluation."
            )
        try:
            installed = version("evalplus")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "humaneval_plus scoring dependency 'evalplus.data' requires "
                "evalplus==0.3.1; install the selected platform requirements."
            ) from error
        if installed != "0.3.1":
            raise RuntimeError(
                f"HumanEval+ requires evalplus==0.3.1, found {installed}; "
                "install the selected platform requirements."
            )
    for name, attributes in TASK_DEPENDENCIES[task].items():
        try:
            module = import_module(name)
            for attr in attributes:
                if not callable(getattr(module, attr)):
                    raise ImportError(f"{name}.{attr} is not callable")
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"{task} scoring dependency {name!r} is unavailable or incompatible; "
                "install the requirements for the selected platform before evaluation."
            ) from error
