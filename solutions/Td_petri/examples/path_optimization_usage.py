"""
路径优化使用示例

演示如何使用优化后的路径查找功能
"""

import numpy as np
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.utils import get_token_path_from_registry, get_next_stage_for_token


def example_1_get_token_path():
    """示例1：使用 get_token_path 方法获取 token 的路径"""
    print("=" * 60)
    print("示例1：动态获取 token 路径")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    # 获取 LP1 和 LP2 的 token
    lp1_idx = net.idle_idx['start'][1]
    lp2_idx = net.idle_idx['start'][0]
    
    # 获取第一个 LP1 token
    if net.marks[lp1_idx].tokens:
        token_lp1 = list(net.marks[lp1_idx].tokens)[0]
        path = net.get_token_path(token_lp1)
        
        print(f"\nLP1 Token (type={token_lp1.type}, where={token_lp1.where}):")
        print(f"  剩余路径阶段数: {len(path)}")
        if path:
            print(f"  下一阶段选项数: {len(path[0])}")
    
    # 获取第一个 LP2 token
    if net.marks[lp2_idx].tokens:
        token_lp2 = list(net.marks[lp2_idx].tokens)[0]
        path = net.get_token_path(token_lp2)
        
        print(f"\nLP2 Token (type={token_lp2.type}, where={token_lp2.where}):")
        print(f"  剩余路径阶段数: {len(path)}")
        if path:
            print(f"  下一阶段选项数: {len(path[0])}")


def example_2_use_utility_functions():
    """示例2：使用工具函数获取路径信息"""
    print("\n" + "=" * 60)
    print("示例2：使用工具函数")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    lp1_idx = net.idle_idx['start'][1]
    
    if net.marks[lp1_idx].tokens:
        token = list(net.marks[lp1_idx].tokens)[0]
        
        # 使用工具函数获取下一阶段
        next_stage = get_next_stage_for_token(
            token, 
            net.path_registry, 
            net.id2t_name
        )
        
        if next_stage:
            print(f"\nToken 的下一阶段:")
            print(f"  阶段有 {len(next_stage)} 个选项")
            print(f"  第一个选项包含 {len(next_stage[0])} 个 transitions")


def example_3_memory_comparison():
    """示例3：内存使用对比"""
    print("\n" + "=" * 60)
    print("示例3：内存优化效果")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    # 统计 token 数量
    total_tokens = 0
    for place in net.marks:
        total_tokens += len(place.tokens)
    
    print(f"\n总 token 数量: {total_tokens}")
    print(f"\n优化前: 每个 token 存储完整路径（约 6-7 个阶段）")
    print(f"优化后: 路径存储在 PathRegistry 中，token 只需 type 和 where 属性")
    print(f"\n内存节省: 显著减少（特别是 token 数量多时）")


def example_4_dynamic_path_update():
    """示例4：演示路径如何随 where 变化"""
    print("\n" + "=" * 60)
    print("示例4：动态路径更新")
    print("=" * 60)
    
    net = TimedPetri()
    obs, mask = net.reset()
    
    # 模拟 token 的 where 变化
    class MockToken:
        def __init__(self, token_type, where):
            self.type = token_type
            self.where = where
    
    # Route C (type=2) 的不同阶段
    print("\nRoute C (type=2) 在不同 where 位置的剩余路径:")
    for where in range(7):
        token = MockToken(token_type=2, where=where)
        path = net.get_token_path(token)
        print(f"  where={where}: 剩余 {len(path)} 个阶段")


if __name__ == '__main__':
    print("路径优化功能演示\n")
    
    example_1_get_token_path()
    example_2_use_utility_functions()
    example_3_memory_comparison()
    example_4_dynamic_path_update()
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
