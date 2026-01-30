"""
PySide6 可视化入口
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from solutions.PPO.enviroment import Env_PN

from .petri_adapter import PetriAdapter
from .viewmodel import PetriViewModel
from .main_window import PetriMainWindow


def build_adapter(adapter_name: str) -> PetriAdapter:
    if adapter_name != "petri":
        raise ValueError(f"不支持的适配器: {adapter_name}")
    env = Env_PN(detailed_reward=True)
    return PetriAdapter(env)


def main() -> int:
    parser = argparse.ArgumentParser(description="PySide6 Petri 可视化")
    parser.add_argument("--adapter", default="petri", choices=["petri"], help="算法适配器")
    args = parser.parse_args()

    adapter = build_adapter(args.adapter)
    viewmodel = PetriViewModel(adapter)

    app = QApplication(sys.argv)
    window = PetriMainWindow(viewmodel)
    window.show()

    # 初始化状态
    viewmodel.reset()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
