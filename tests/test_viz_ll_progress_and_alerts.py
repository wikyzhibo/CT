from types import SimpleNamespace

import pytest

from visualization.petri_single_adapter import PetriSingleAdapter


def test_adapter_time_to_scrap_for_ll_uses_three_times_p_residual() -> None:
    adapter = PetriSingleAdapter.__new__(PetriSingleAdapter)
    adapter.net = SimpleNamespace(P_Residual_time=10, D_Residual_time=5)

    lld = SimpleNamespace(type=5, name="LLD", processing_time=75)
    llc = SimpleNamespace(type=5, name="LLC", processing_time=0)
    other_buffer = SimpleNamespace(type=5, name="BUF_X", processing_time=50)

    assert adapter._time_to_scrap(lld, 100.0) == 5.0
    assert adapter._time_to_scrap(llc, 15.0) == 15.0
    assert adapter._time_to_scrap(other_buffer, 15.0) == -1.0


def test_chamber_widget_treats_type5_with_proc_time_as_timed_chamber() -> None:
    pytest.importorskip("PySide6")
    from visualization.widgets.chamber_widget import ChamberWidget

    ll_timed = SimpleNamespace(place_type=5, proc_time=65.0)
    ll_not_timed = SimpleNamespace(place_type=5, proc_time=0.0)
    pm_timed = SimpleNamespace(place_type=1, proc_time=120.0)
    transport = SimpleNamespace(place_type=2, proc_time=5.0)

    assert ChamberWidget._is_timed_chamber_wafer(ll_timed)
    assert not ChamberWidget._is_timed_chamber_wafer(ll_not_timed)
    assert ChamberWidget._is_timed_chamber_wafer(pm_timed)
    assert not ChamberWidget._is_timed_chamber_wafer(transport)
