
import math

class TDPNParser:
    """
    Petri 网动作解析器
    负责将离散的时间事件流映射并离散化为 Petri 网的并发动作序列。
    """
    def __init__(self):
        # Petri 网仿真的步长时间间隔（秒）
        self.pn_interval = 5
    
    def parse(self, events):
        """
        将事件列表解析为 PN 的 (a1, a2) 变迁名称序列。
        
        参数:
            events: 事件列表，格式为 (time, arm, kind, from_loc, to_loc)
        返回:
            sequence: 字典列表，包含 {"step": 步数, "time": 时间戳, "actions": [TM2动作, TM3动作]}
        """
        # 1. 映射事件：将外部事件转换为 Petri 网内部的变迁和对应的机器人
        mapped_events = []
        for t, arm, kind, from_loc, to_loc in events:
            pn_action, robot = self._map_action_v2(arm, kind, from_loc, to_loc)
            if pn_action:
                mapped_events.append((t, pn_action, robot))
        
        if not mapped_events:
            return []

        # 2. 按 5s 窗口进行分组离散化
        max_time = max(e[0] for e in mapped_events)
        duration = int(math.ceil(max_time / self.pn_interval)) + 2
        
        sequence = []
        # 按发生时间排序
        mapped_events.sort(key=lambda x: x[0])
        
        current_event_idx = 0
        n_events = len(mapped_events)
        
        # 遍历每个时间窗，填充动作
        for i in range(duration):
            time_start = i * self.pn_interval
            time_end = (i + 1) * self.pn_interval
            
            tm2_act = None
            tm3_act = None
            
            # 将落在该时间窗内的事件提取出来
            while current_event_idx < n_events:
                t, act, robot = mapped_events[current_event_idx]
                
                # 如果事件时间超过当前窗，停止匹配
                if t >= time_end:
                    break
                # 时间对齐逻辑：允许微量的时间提前/滞后容差
                if t >= time_start or (t > time_start - 2): 
                    if robot == "TM2":
                        if tm2_act is None:
                            tm2_act = act
                        else:
                            # 单个时间窗内 TM2 只能执行一个动作
                            break 
                    elif robot == "TM3":
                        if tm3_act is None:
                            tm3_act = act
                        else:
                            # 单个时间窗内 TM3 只能执行一个动作
                            break
                
                current_event_idx += 1
            
            # 记录当前步的并发动作对
            sequence.append({
                "step": i,
                "time": time_start,
                "actions": [tm2_act, tm3_act]
            })
            
            if current_event_idx >= n_events:
                break
        
        # 修剪序列开头的纯等待步骤，以加快验证速度
        while sequence and sequence[0]["actions"] == [None, None]:
            sequence.pop(0)
        # 修正序列时间
        for i in range(len(sequence)):
            sequence[i]["step"] = i + 1
            sequence[i]["time"] = i * self.pn_interval + 6

        return sequence

    def _map_action_v2(self, arm, kind, from_loc, to_loc):
        """
        核心映射逻辑：根据机械臂编号、动作类型、库所来源和去向映射为 PN 变迁。
        kind: 0=PICK (取), 1=LOAD (放)
        返回: (PN变迁名称, 机器人名称)
        """
        # 标准化位置名称
        from_loc = from_loc.upper()
        if not to_loc: to_loc = "" # 处理空去向
        to_loc = to_loc.upper()
        
        # --- Route 1 (LP1) 的映射规则 ---
        # TM2 机械臂相关的动作 (arm == 2)
        if arm == 2:
            # 1. 从 LLA_S2 取到 PM7/8 (s1)
            if "LLA_S2" in from_loc and ("PM7" in to_loc or "PM8" in to_loc):
                if kind == 0: return "u_LP1_s1", "TM2"
                if kind == 1: return "t_s1", "TM2"
            
            # 2. 从 PM7/8 (s1) 取到 LLC (s2)
            if ("PM7" in from_loc or "PM8" in from_loc) and "LLC" in to_loc:
                if kind == 0: return "u_s1_s2", "TM2" 
                if kind == 1: return "t_s2", "TM2"
            
            # 3. 从 LLD (s4) 取到 PM9/10 (s5)
            if "LLD" in from_loc and ("PM9" in to_loc or "PM10" in to_loc):
                if kind == 0: return "u_s4_s5", "TM2"
                if kind == 1: return "t_s5", "TM2"
            
            # 4. 从 PM9/10 (s5) 取到 LLB (LP_done)
            if ("PM9" in from_loc or "PM10" in from_loc) and "LLB" in to_loc:
                if kind == 0: return "u_s5_LP_done", "TM2"
                if kind == 1: return "t_LP_done", "TM2"

        # TM3 机械臂相关的动作 (arm == 3)
        if arm == 3:
            # 1. 从 LLC (s2) 取到 PM1-4 (s3)
            if "LLC" in from_loc and "PM" in to_loc:
                if any(x in to_loc for x in ["PM1", "PM2", "PM3", "PM4"]):
                    if kind == 0: return "u_s2_s3", "TM3"
                    if kind == 1: return "t_s3", "TM3"
            
            # 2. 从 PM1-4 (s3) 取到 LLD (s4)
            if "PM" in from_loc and "LLD" in to_loc:
                if any(x in from_loc for x in ["PM1", "PM2", "PM3", "PM4"]):
                    if kind == 0: return "u_s3_s4", "TM3"
                    if kind == 1: return "t_s4", "TM3"
                    
        return None, None
    
    def _map_action(self, t_name):
        # 旧版映射逻辑（已弃用）
        return None, None
