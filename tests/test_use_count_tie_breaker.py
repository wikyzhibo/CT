import unittest

from solutions.A.petri_net import ClusterTool


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


if __name__ == "__main__":
    unittest.main()
