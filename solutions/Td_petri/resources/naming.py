"""
变迁命名解析工具。

提供从变迁名解析资源名的工具函数。
"""

from typing import List


# 默认模块列表
DEFAULT_MODULE_LIST = ['PM7', 'PM8', 'PM1', 'PM2', 'PM3', 'PM4', 'LLC', 'LLD', 'PM9', 'PM10']


def get_transition_resources(t_name: str) -> List[str]:
    """
    从变迁名解析资源名。
    
    V3 命名规则：
    - 加工：PROC__PM7 -> ["PM7"], PROC__LLA_S2 -> ["LLA_S2"]
    - 搬运：ARM2_PICK__A__TO__B -> ["ARM2"]
    - 其他：[]
    
    Args:
        t_name: 变迁名称
    
    Returns:
        资源名列表
        
    Example:
        >>> get_transition_resources("PROC__PM7")
        ["PM7"]
        >>> get_transition_resources("ARM2_PICK__LLA_S2__TO__PM7")
        ["ARM2"]
    """
    # 工艺加工：占用对应模块资源
    if t_name.startswith("PROC__"):
        mod = t_name.split("__", 1)[1]
        return [mod]
    
    # 搬运动作：占用机械手资源
    if t_name.startswith("ARM"):
        arm = t_name.split("_", 1)[0]  # "ARM2"
        return [arm]
    
    return []


def get_close_resources(t_name: str, module_list: List[str] = None) -> str:
    """
    从变迁名解析需要关闭的资源。
    
    只有 PICK 动作会关闭资源（从腔室取出晶圆时释放腔室）。
    
    Args:
        t_name: 变迁名称
        module_list: 有效模块列表（默认使用 DEFAULT_MODULE_LIST）
    
    Returns:
        需要关闭的资源名，如果没有则返回空字符串
        
    Example:
        >>> get_close_resources("ARM2_PICK__PM7__TO__LLC")
        "PM7"
        >>> get_close_resources("ARM2_LOAD__LLC__TO__PM7")
        ""
    """
    if module_list is None:
        module_list = DEFAULT_MODULE_LIST
    
    # 只有 ARM_PICK 动作会关闭资源
    if t_name.startswith("ARM") and "__" in t_name:
        parts = t_name.split("__")
        # parts = ['ARM2_PICK', 'LLA_S2', 'TO', 'PM7']
        sub_part = parts[0].split("_")
        if len(sub_part) > 1 and sub_part[1] != 'PICK':
            return ""
        from_chamber = parts[1]
        if from_chamber in module_list:
            return from_chamber
    
    return ""
