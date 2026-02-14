"""
甘特图可视化使用示例

演示如何使用重构后的可视化模块绘制甘特图
"""

from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.visualization import GanttRenderer, render_gantt_from_petri


def example_1_using_method():
    """示例1：使用TimedPetri的render_gantt方法（向后兼容）"""
    print("=" * 60)
    print("示例1：使用render_gantt方法")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    # 运行几步
    for _ in range(10):
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            break
        action = valid_actions[0]
        mask, obs, time, done, reward = net.step(action)
        if done:
            break
    
    # 使用原有方法绘制甘特图
    net.render_gantt(out_path='../../results/', policy=1, with_label=True, no_arm=True)
    print("甘特图已保存到 results/ 目录")


def example_2_using_renderer():
    """示例2：使用GanttRenderer类"""
    print("\n" + "=" * 60)
    print("示例2：使用GanttRenderer类")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    # 运行几步
    for _ in range(10):
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            break
        action = valid_actions[0]
        mask, obs, time, done, reward = net.step(action)
        if done:
            break
    
    # 使用GanttRenderer
    renderer = GanttRenderer(net)
    renderer.render(out_path='../../results/', policy=2, with_label=True, no_arm=False)
    print("甘特图已保存到 results/ 目录")


def example_3_using_function():
    """示例3：使用便捷函数"""
    print("\n" + "=" * 60)
    print("示例3：使用便捷函数")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    # 运行几步
    for _ in range(10):
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            break
        action = valid_actions[0]
        mask, obs, time, done, reward = net.step(action)
        if done:
            break
    
    # 使用便捷函数
    render_gantt_from_petri(net, out_path='../../results/', policy=3)
    print("甘特图已保存到 results/ 目录")


if __name__ == '__main__':
    import numpy as np
    
    print("甘特图可视化示例\n")
    
    # 运行示例
    example_1_using_method()
    example_2_using_renderer()
    example_3_using_function()
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
