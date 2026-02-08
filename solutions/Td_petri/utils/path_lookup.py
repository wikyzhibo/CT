"""
路径查找工具模块

提供根据 token 属性动态查找路径的辅助函数
"""

from typing import List


def get_token_path_from_registry(token, path_registry, id2t_name):
    """
    根据 token 的 type 和 where 属性从 PathRegistry 获取路径
    
    这是一个独立的工具函数，可以在不修改 TimedPetri 类的情况下使用。
    
    Args:
        token: WaferToken 实例，包含 type 和 where 属性
        path_registry: PathRegistry 实例
        id2t_name: transition ID 到名称的映射列表
    
    Returns:
        从当前位置开始的剩余路径（transition ID 列表的列表）
    
    示例:
        >>> from solutions.Td_petri.rl import PathRegistry
        >>> registry = PathRegistry()
        >>> path = get_token_path_from_registry(token, registry, id2t_name)
    """
    # 根据 token.type 确定路径类型
    # type=1: Route D (LP1)
    # type=2: Route C (LP2)
    if token.type == 1:
        route = 'D'
    elif token.type == 2:
        route = 'C'
    else:
        return []
    
    # 获取完整路径
    full_path = path_registry.get_path_indices(id2t_name, route)
    
    # 根据 where 返回剩余路径
    where = getattr(token, 'where', 0)
    return full_path[where:]


def get_next_stage_for_token(token, path_registry, id2t_name):
    """
    获取 token 的下一个阶段（stage）
    
    Args:
        token: WaferToken 实例
        path_registry: PathRegistry 实例
        id2t_name: transition ID 到名称的映射
    
    Returns:
        下一个阶段的 transition ID 列表，如果没有则返回 None
    
    示例:
        >>> next_stage = get_next_stage_for_token(token, registry, id2t_name)
        >>> if next_stage:
        >>>     print(f"Next stage has {len(next_stage)} options")
    """
    remaining_path = get_token_path_from_registry(token, path_registry, id2t_name)
    
    if not remaining_path:
        return None
    
    # 返回下一个阶段（第一个元素）
    return remaining_path[0]


def get_all_remaining_stages(token, path_registry, id2t_name):
    """
    获取 token 的所有剩余阶段
    
    Args:
        token: WaferToken 实例
        path_registry: PathRegistry 实例
        id2t_name: transition ID 到名称的映射
    
    Returns:
        所有剩余阶段的列表
    """
    return get_token_path_from_registry(token, path_registry, id2t_name)
