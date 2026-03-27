"""solutions 顶层导入入口。"""

from importlib import import_module

_PACKAGE_EXPORTS = {
    "A": ".A",
    "B": ".B",
}

_MODULE_EXPORTS = {
    "clustertool_config": ".B.clustertool_config",
    "core": ".B.core",
    "deprecated": ".A.deprecated",
    "Env": ".B.Env",
    "eval": ".A.eval",
    "model_builder": ".A.model_builder",
    "parse_sequences": ".B.parse_sequences",
    "petri_net": ".A.petri_net",
    "plot_train_metrics": ".B.plot_train_metrics",
    "pn_models": ".B.pn_models",
    "ppo_models": ".B.ppo_models",
    "ppo_trainer": ".A.ppo_trainer",
    "rl_env": ".A.rl_env",
    "run_pdr": ".B.run_pdr",
    "takt_analysis": ".A.takt_analysis",
    "train": ".B.train",
    "training_config": ".B.training_config",
    "utils": ".A.utils",
    "validation": ".B.validation",
}

__all__ = [*_PACKAGE_EXPORTS, *_MODULE_EXPORTS]


def __getattr__(name: str):
    target = _PACKAGE_EXPORTS.get(name) or _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
