import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from solutions.A.rl_env import Env_PN_Single
from visualization.graphics.wafer_item import WaferItem
from visualization.petri_single_adapter import PetriSingleAdapter
from visualization.theme import ColorTheme
from visualization.widgets.chamber_widget import ChamberWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_adapter(route_name: str) -> PetriSingleAdapter:
    env = Env_PN_Single(single_route_name=route_name)
    return PetriSingleAdapter(env, step_verbose=False)


def _action_id(adapter: PetriSingleAdapter, action_name: str) -> int:
    for action in adapter.get_enabled_actions():
        if action.action_name == action_name:
            return int(action.action_id)
    raise AssertionError(f"missing action: {action_name}")


def test_cascade_single_reset_maps_start_end_and_tm1():
    adapter = _build_adapter("2-1")
    state = adapter.reset()

    assert [c.name for c in state.start_buffers] == ["LP"]
    assert [c.name for c in state.end_buffers] == ["LP_done"]
    assert set(state.robot_states.keys()) == {"TM1", "TM2", "TM3"}

    chamber_names = [c.name for c in state.chambers]
    assert len(chamber_names) == len(set(chamber_names))
    assert "TM1" not in chamber_names

    lp = next(c for c in state.chambers if c.name == "LP")
    assert lp.wafers
    assert not any(c.name == "LLA" and c.wafers for c in state.chambers)


def test_tm1_pick_shows_wafer_on_robot_not_fake_tm1_chamber():
    adapter = _build_adapter("2-1")
    adapter.reset()

    state, _, _, _ = adapter.step(_action_id(adapter, "u_LP1_TM1"))

    assert [w.place_name for w in state.robot_states["TM1"].wafers] == ["TM1"]
    assert not any(c.name == "TM1" for c in state.chambers)


def test_al_timed_buffer_uses_non_scrap_rendering(qapp):
    adapter = _build_adapter("2-1")
    adapter.reset()

    for action_name in ("u_LP1_TM1", "t_TM1_AL", "WAIT_5s", "WAIT_5s"):
        state, _, _, _ = adapter.step(_action_id(adapter, action_name))

    al = next(c for c in state.chambers if c.name == "AL")
    wafer = al.wafers[0]

    assert wafer.time_to_scrap < 0

    theme = ColorTheme()
    wafer_item = WaferItem(theme)
    chamber_widget = ChamberWidget(al, theme)

    assert wafer_item._get_wafer_color(wafer) == theme.warning
    assert wafer_item._timed_wafer_main_text(wafer) != "SCRAP"
    assert wafer_item._timed_wafer_main_text(wafer).startswith("+")
    assert chamber_widget._get_wafer_color(wafer) == theme.warning


def test_legacy_route_still_uses_lp_and_lp_done_mapping():
    adapter = _build_adapter("1-1")
    state = adapter.reset()

    assert [c.name for c in state.start_buffers] == ["LP"]
    assert [c.name for c in state.end_buffers] == ["LP_done"]
    assert any(c.name == "LP" and c.wafers for c in state.chambers)
    assert not any(c.name == "LLA" and c.wafers for c in state.chambers)
