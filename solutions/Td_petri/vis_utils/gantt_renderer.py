"""
Gantt图可视化模块

这个模块提供了Timed Petri Net的甘特图可视化功能，
从主文件tdpn.py中独立出来，便于维护和复用。
"""

from typing import Dict, List, Tuple
from visualization.plot import plot_gantt_hatched_residence, Op


def res_occ_to_ops(res_occ: dict, proc: dict) -> List[Op]:
    """
    将 net.res_occ 转换为 Op 列表
    - 工艺资源：PM / LLC / LLD（is_arm=False）
    - 机械手资源：ARM2 / ARM3（is_arm=True，甘特图画成绿色）
    """

    ops: List[Op] = []

    def map_stage_machine(res: str) -> Tuple[int, int, bool]:
        # -------- 工艺 --------
        if res in ("PM7", "PM8"):
            return 1, 0 if res == "PM7" else 1, False
        if res == "LLC":
            return 2, 0, False
        if res.startswith("PM") and res[2:].isdigit() and 1 <= int(res[2:]) <= 4:
            return 3, int(res[2:]) - 1, False
        if res == "LLD":
            return 4, 0, False
        if res in ("PM9", "PM10"):
            return 5, 0 if res == "PM9" else 1, False

        # -------- 机械手 --------
        if res == "ARM2":
            return 6, 0, True
        if res == "ARM3":
            return 7, 0, True

        return -1, 0, False

    for res_name, intervals in res_occ.items():
        stage, machine, is_arm = map_stage_machine(res_name)
        if stage < 0:
            continue

        for iv in intervals:
            proc_end = iv.end
            if stage > 0:
                proc_end = proc[stage]
            ops.append(
                Op(
                    job=int(iv.tok_key),
                    stage=int(stage),
                    machine=int(machine),
                    start=float(iv.start),
                    proc_end=float(iv.start + proc_end),
                    end=float(iv.end),
                    is_arm=is_arm,
                    kind=iv.kind,
                    from_loc=getattr(iv, 'from_loc', ''),
                    to_loc=getattr(iv, 'to_loc', '')
                )
            )

    return ops


class GanttRenderer:
    """
    甘特图渲染器
    
    封装了甘特图绘制的所有逻辑，使得TimedPetri类更加简洁。
    
    Example:
        >>> renderer = GanttRenderer(net)
        >>> renderer.render(out_path='results/', policy=1)
    """
    
    def __init__(self, petri_net):
        """
        初始化渲染器
        
        Args:
            petri_net: TimedPetri实例
        """
        self.net = petri_net
    
    def render(self, out_path: str = "../../results/", 
               policy: int = 1, 
               with_label: bool = True, 
               no_arm: bool = True) -> None:
        """
        渲染甘特图
        
        Args:
            out_path: 输出路径
            policy: 策略编号（用于文件命名）
            with_label: 是否显示标签
            no_arm: 是否隐藏机械手操作
        """
        # 转换资源占用为操作列表
        ops = res_occ_to_ops(self.net.res_occ, self.net.proc)
        
        # 绘制甘特图
        plot_gantt_hatched_residence(
            ops=ops,
            proc_time=self.net.proc,
            capacity=self.net.stage_c,
            n_jobs=self.net.n_wafer,
            out_path=out_path,
            with_label=with_label,
            arm_info={},
            policy=policy,
            no_arm=no_arm
        )
    
    def render_with_custom_ops(self, ops: List[Op], 
                               out_path: str = "../../results/",
                               policy: int = 1,
                               with_label: bool = True,
                               no_arm: bool = True) -> None:
        """
        使用自定义操作列表渲染甘特图
        
        Args:
            ops: 自定义操作列表
            out_path: 输出路径
            policy: 策略编号
            with_label: 是否显示标签
            no_arm: 是否隐藏机械手操作
        """
        plot_gantt_hatched_residence(
            ops=ops,
            proc_time=self.net.proc,
            capacity=self.net.stage_c,
            n_jobs=self.net.n_wafer,
            out_path=out_path,
            with_label=with_label,
            arm_info={},
            policy=policy,
            no_arm=no_arm
        )


# 便捷函数：直接从TimedPetri实例渲染甘特图
def render_gantt_from_petri(petri_net, 
                            out_path: str = "../../results/",
                            policy: int = 1,
                            with_label: bool = True,
                            no_arm: bool = True) -> None:
    """
    从TimedPetri实例直接渲染甘特图（便捷函数）
    
    Args:
        petri_net: TimedPetri实例
        out_path: 输出路径
        policy: 策略编号
        with_label: 是否显示标签
        no_arm: 是否隐藏机械手操作
    
    Example:
        >>> from solutions.Td_petri.tdpn import TimedPetri
        >>> from solutions.Td_petri.visualization import render_gantt_from_petri
        >>> net = TimedPetri()
        >>> # ... 运行仿真 ...
        >>> render_gantt_from_petri(net, out_path='results/')
    """
    renderer = GanttRenderer(petri_net)
    renderer.render(out_path, policy, with_label, no_arm)
