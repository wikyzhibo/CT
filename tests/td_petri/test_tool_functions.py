"""
Timed Petri Net 工具函数的单元测试。

验证底层资源管理和时间轴计算逻辑。
"""

import pytest
from solutions.Td_petri.resources.interval_utils import (
    Interval,
    _first_free_time_at,
    _first_free_time_open,
    _insert_interval_sorted
)
from solutions.Td_petri.resources.resource_manager import ResourceManager

class TestIntervalUtils:
    """测试 interval_utils.py 中的工具函数"""

    def test_insert_interval_sorted(self):
        """测试排序插入区间"""
        occ = []
        i1 = Interval(start=10, end=20, tok_key=1)
        i2 = Interval(start=30, end=40, tok_key=2)
        i3 = Interval(start=22, end=28, tok_key=3)
        
        _insert_interval_sorted(occ, i1)
        _insert_interval_sorted(occ, i2)
        _insert_interval_sorted(occ, i3)
        
        assert len(occ) == 3
        # 验证排序: 10-20, 22-28, 30-40
        assert occ[0].start == 10
        assert occ[1].start == 22
        assert occ[2].start == 30

    def test_first_free_time_at(self):
        """测试在指定长度内寻找空闲时间"""
        # 已占用区间: [10, 20], [30, 40]
        occ = [
            Interval(start=10, end=20),
            Interval(start=30, end=40)
        ]
        
        # 1. 在 0 点寻找长度 5 的空位 -> 应该返回 0 (0-5 不冲突)
        res1 = _first_free_time_at(occ, 0, 5)
        print(f"DEBUG 1: {res1}")
        assert res1 == 0
        
        # 2. 在 8 点寻找长度 5 的空位 -> 应该推迟到 20 
        # 因为 8-13 涵盖了 10-13 (被 [10,20] 占用)
        res2 = _first_free_time_at(occ, 8, 5)
        print(f"DEBUG 2: {res2}")
        assert res2 == 20
        
        # 3. 在 22 点寻找长度 5 的空位 -> 应该返回 22 (22-27 不超过 30)
        res3 = _first_free_time_at(occ, 22, 5)
        print(f"DEBUG 3: {res3}")
        assert res3 == 22
        
        # 4. 在 22 点寻找长度 10 的空位 -> 应该推迟到 40 
        # 因为 22-32 涵盖了 30-32 (被 [30,40] 占用)
        res4 = _first_free_time_at(occ, 22, 10)
        print(f"DEBUG 4: {res4}")
        assert res4 == 40

    def test_first_free_time_open(self):
        """测试寻找资源的第一个开放时间点（之后永久空闲）"""
        # 已占用区间: [10, 20], [30, 40]
        occ = [
            Interval(start=10, end=20),
            Interval(start=30, end=40)
        ]
        
        # 1. 从 0 点开始找 -> 应该返回 0 (0 点可开始且不冲突) - 注意：这个函数逻辑是找不覆盖已有区间的点
        # 实际逻辑：如果 start 在某个区间内，或者 start 之后有区间，则需要跳过。
        # _first_free_time_open 通常用于 PROC 这种一旦占用就持续到 INF 的情况，或者是找最后一个释放点。
        
        # 假设逻辑是寻找 t >= start 且 [t, INF] 不与 occ 冲突的点
        assert _first_free_time_open(occ, 0) == 40
        assert _first_free_time_open(occ, 35) == 40
        assert _first_free_time_open(occ, 50) == 50

class TestResourceManager:
    """测试 ResourceManager"""

    def test_sync_start(self):
        """测试多资源同步对齐"""
        rm = ResourceManager()
        # R1 占用 [10, 20]
        rm.res_occ["R1"] = [Interval(start=10, end=20)]
        # R2 占用 [15, 25]
        rm.res_occ["R2"] = [Interval(start=15, end=25)]
        
        # 同时申请 R1, R2 长度 5，从 0 开始
        # 0-5 平安
        assert rm.sync_start(["R1", "R2"], 0, 5) == 0
        
        # 从 8 开始申请长度 5
        # R1 8-13 冲突 -> 推迟到 20
        # R2 20-25 冲突 -> 推迟到 25
        # 最终应该是 25
        assert rm.sync_start(["R1", "R2"], 8, 5) == 25

    def test_reset(self):
        """测试重置功能"""
        rm = ResourceManager()
        rm.res_occ["R1"] = [Interval(start=10, end=20)]
        rm.open_mod_occ[("M1", 1)] = Interval(start=10, end=20)
        
        rm.reset()
        assert len(rm.res_occ) == 0
        assert len(rm.open_mod_occ) == 0

if __name__ == "__main__":
    pytest.main([__file__])
