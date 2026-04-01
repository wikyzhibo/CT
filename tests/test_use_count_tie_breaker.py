import unittest
from collections import deque
from types import MethodType, SimpleNamespace

import numpy as np

from solutions.A.petri_net import ClusterTool
import visualization.plot as plot_module


class TestUseCountTieBreaker(unittest.TestCase):
    def _new_tool(self) -> ClusterTool:
        tool = ClusterTool.__new__(ClusterTool)
        tool._place_use_count = {"PM9": 0, "PM10": 0}
        return tool

    def test_tie_breaker_is_stable_without_state_change(self) -> None:
        tool = self._new_tool()
        picks = [
            tool._select_min_use_count_target(("PM9", "PM10"), -1),
            tool._select_min_use_count_target(("PM9", "PM10"), -1),
            tool._select_min_use_count_target(("PM9", "PM10"), -1),
            tool._select_min_use_count_target(("PM9", "PM10"), -1),
        ]
        self.assertEqual(picks, ["PM9", "PM9", "PM9", "PM9"])

    def test_tie_breaker_rotates_after_each_use_count_update(self) -> None:
        tool = self._new_tool()
        picks = []
        for _ in range(4):
            picked = tool._select_min_use_count_target(("PM9", "PM10"), -1)
            picks.append(picked)
            tool._place_use_count[picked] += 1
        self.assertEqual(picks, ["PM9", "PM10", "PM9", "PM10"])

    def test_prefers_lower_use_count_before_tie_break(self) -> None:
        tool = self._new_tool()
        tool._place_use_count["PM9"] = 3
        tool._place_use_count["PM10"] = 1
        picked = tool._select_min_use_count_target(("PM9", "PM10"), -1)
        self.assertEqual(picked, "PM10")

    def test_three_way_targets_follow_fixed_stage_order_cycle(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tool._place_use_count = {"PM2": 0, "PM3": 0, "PM5": 0}
        picks = []
        for _ in range(6):
            picked = tool._select_min_use_count_target(("PM2", "PM3", "PM5"), -1)
            picks.append(picked)
            tool._place_use_count[picked] += 1
        self.assertEqual(picks, ["PM2", "PM3", "PM5", "PM2", "PM3", "PM5"])


class TestSharedRatioReleaseCycle(unittest.TestCase):
    def test_build_release_ratio_cycle(self) -> None:
        cycle = ClusterTool._build_release_ratio_cycle([1, 2, 1])
        self.assertEqual(cycle, (1, 2, 2, 3))

    def test_required_release_type_uses_cycle_index(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tool._shared_ratio_cycle_enabled = True
        tool._shared_ratio_cycle_types = (1, 2, 2)
        tool._shared_ratio_cycle_idx = 1
        self.assertEqual(tool._required_release_type(), 2)

    def test_advance_release_ratio_cycle_wraps(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tool._shared_ratio_cycle_enabled = True
        tool._shared_ratio_cycle_types = (1, 2, 2)
        tool._shared_ratio_cycle_idx = 2
        tool._advance_release_ratio_cycle()
        self.assertEqual(tool._shared_ratio_cycle_idx, 0)

    def test_advance_release_ratio_cycle_noop_when_disabled(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tool._shared_ratio_cycle_enabled = False
        tool._shared_ratio_cycle_types = ()
        tool._shared_ratio_cycle_idx = 0
        tool._advance_release_ratio_cycle()
        self.assertEqual(tool._shared_ratio_cycle_idx, 0)


class TestWaitKeyEventTruncation(unittest.TestCase):
    @staticmethod
    def _place(name: str, *, is_dtm: bool, place_type: int, ptime: int, stays: list[int]):
        return SimpleNamespace(
            name=name,
            is_dtm=is_dtm,
            type=place_type,
            processing_time=ptime,
            tokens=deque(SimpleNamespace(stay_time=s) for s in stays),
        )

    def test_get_next_event_delta_excludes_tm_transport_event(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tm2 = self._place("TM2", is_dtm=True, place_type=2, ptime=5, stays=[0])
        pm1 = self._place("PM1", is_dtm=False, place_type=1, ptime=8, stays=[0])
        lp1 = self._place("LP1", is_dtm=False, place_type=3, ptime=0, stays=[-9])
        tool.marks = [tm2, pm1, lp1]
        tool._load_port_names = ("LP1",)
        tool._place_by_name = {"TM2": tm2, "PM1": pm1, "LP1": lp1}
        tool.stride = True

        delta = tool.get_next_event_delta()

        self.assertEqual(delta, 5)

    def _make_wait_tool(self) -> tuple[ClusterTool, dict]:
        tool = ClusterTool.__new__(ClusterTool)
        tool.T = 0
        tool.wait_durations = [5, 50]
        tool._lp_done = SimpleNamespace(tokens=deque())
        tool.n_wafer = 99
        tool.time = 0
        tool.MAX_TIME = 10_000
        tool._last_deadlock = False
        tool.stride = True
        tool._consecutive_wait_time = 0
        tool.idle_timeout = 1_000
        tool._idle_penalty_applied = False
        tool.idle_event_penalty = 0
        tool.finish_event_reward = 0
        tool.scrap_event_penalty = 0
        tool._per_wafer_reward = 0.0
        captured: dict[str, int] = {}

        def _advance_and_compute_reward(self, dt: int, t1: int, t2: int):
            captured["dt"] = int(dt)
            return 0.0, {"is_scrap": False, "scrap_info": None}

        tool._advance_and_compute_reward = MethodType(_advance_and_compute_reward, tool)
        tool._should_cancel_resident_scrap_after_fire = MethodType(lambda self, scan, log_entry: False, tool)
        tool.get_action_mask = MethodType(lambda self, wait_action_start=None, n_actions=None: np.array([], dtype=bool), tool)
        tool.get_obs = MethodType(lambda self: np.array([], dtype=np.float32), tool)
        return tool, captured

    def test_step_truncates_to_5s_when_ready_chamber_exists(self) -> None:
        tool, captured = self._make_wait_tool()

        def _next_event(self):
            return 5

        tool.get_next_event_delta = MethodType(_next_event, tool)
        tool.step(wait_duration=50)
        self.assertEqual(captured["dt"], 5)

    def test_step_truncates_to_5s_when_tm_holds_wafer(self) -> None:
        tool, captured = self._make_wait_tool()

        def _next_event(self):
            return 5

        tool.get_next_event_delta = MethodType(_next_event, tool)
        tool.step(wait_duration=50)
        self.assertEqual(captured["dt"], 5)

    def test_step_uses_next_event_delta_when_no_important_task(self) -> None:
        tool, captured = self._make_wait_tool()

        def _next_event(self):
            self._last_next_event_scan = {"has_ready_chamber": False, "has_tm_holding_wafer": False}
            return 12

        tool.get_next_event_delta = MethodType(_next_event, tool)
        tool.step(wait_duration=50)
        self.assertEqual(captured["dt"], 12)


class TestEvalGanttRecording(unittest.TestCase):
    def test_training_mode_does_not_record_eval_gantt(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tool._training = True
        tool._eval_gantt_place_to_sm = {"PM1": (1, 0)}
        tool._eval_gantt_slots = {"PM1": {}}
        tool._eval_gantt_closed_ops = []

        tool._record_eval_gantt_enter("PM1", token_id=7, start_time=10, proc_time=40)

        self.assertEqual(tool._eval_gantt_slots["PM1"], {})
        self.assertEqual(tool._eval_gantt_closed_ops, [])

    def test_render_gantt_prefers_eval_gantt_records(self) -> None:
        tool = ClusterTool.__new__(ClusterTool)
        tool.single_route_name = "4-13"
        tool._gantt_route_stages = [["PM1"]]
        tool._base_proc_time_map = {"PM1": 30}
        tool._training = False
        tool.time = 100
        tool.fire_log = []
        tool._eval_gantt_place_to_sm = {"PM1": (1, 0)}
        tool._eval_gantt_slots = {"PM1": {}}
        tool._eval_gantt_closed_ops = [
            {
                "job": 7,
                "stage": 1,
                "machine": 0,
                "start": 10.0,
                "proc_end": 50.0,
                "end": 70.0,
            }
        ]

        captured = {}
        original = plot_module.plot_gantt_hatched_residence

        def _fake_plot(**kwargs):
            captured.update(kwargs)

        try:
            plot_module.plot_gantt_hatched_residence = _fake_plot
            tool.render_gantt("results/gantt/eval_only.png")
        finally:
            plot_module.plot_gantt_hatched_residence = original

        self.assertIn("ops", captured)
        self.assertEqual(len(captured["ops"]), 1)
        op = captured["ops"][0]
        self.assertEqual(op.job, 7)
        self.assertEqual(op.proc_end, 50.0)


if __name__ == "__main__":
    unittest.main()
