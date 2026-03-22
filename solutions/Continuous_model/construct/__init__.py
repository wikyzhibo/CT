"""子包 `construct.*` 与同级模块 `construct.py` 共存：从此包 `import BasedToken` 等须转发到 `construct.py`。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_LEGACY = Path(__file__).resolve().parent.parent / "construct.py"
_spec = importlib.util.spec_from_file_location(
    "solutions.Continuous_model._legacy_construct_py",
    _LEGACY,
)
_legacy_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["solutions.Continuous_model._legacy_construct_py"] = _legacy_mod
_spec.loader.exec_module(_legacy_mod)

BasedToken = _legacy_mod.BasedToken
INF = _legacy_mod.INF
ModuleSpec = _legacy_mod.ModuleSpec
RobotSpec = _legacy_mod.RobotSpec
SharedGroup = _legacy_mod.SharedGroup
Stage = _legacy_mod.Stage
SuperPetriBuilder = _legacy_mod.SuperPetriBuilder

__all__ = [
    "BasedToken",
    "INF",
    "ModuleSpec",
    "RobotSpec",
    "SharedGroup",
    "Stage",
    "SuperPetriBuilder",
]
