import pytest

from solutions.Continuous_model.env_single import Env_PN_Single
from visualization.petri_single_adapter import PetriSingleAdapter


@pytest.mark.parametrize("route_code", [1, 2])
def test_viz_pm5_pm6_not_disabled_in_cascade(route_code: int) -> None:
    """
    级联（device_mode="cascade"）下，可视化界面中的 PM5/PM6 不应被强制标记为 disabled。

    说明：
    - cascade 模式下合法 route_code 为 1..6，因此这里覆盖 1 和 2 两个代表值即可。
    - 构网中 PM5/PM6 通常不在 net.marks 里，因此适配器会补占位腔室；占位腔室也不应为 disabled。
    """

    env = Env_PN_Single(
        detailed_reward=False,
        device_mode="cascade",
        robot_capacity=1,
        route_code=route_code,
        # 只提供构网会用到的少量关键键；缺失项会落回 construct_single 的默认值
        process_time_map={"PM7": 70, "LLD": 70, "PM9": 200},
    )
    adapter = PetriSingleAdapter(env, step_verbose=False)
    state = adapter.reset()

    chamber_map = {c.name: c for c in state.chambers}
    assert "PM5" in chamber_map
    assert "PM6" in chamber_map

    assert chamber_map["PM5"].status != "disabled"
    assert chamber_map["PM5"].chamber_type != "disabled"

    assert chamber_map["PM6"].status != "disabled"
    assert chamber_map["PM6"].chamber_type != "disabled"

