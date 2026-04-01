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


if __name__ == "__main__":
    unittest.main()
