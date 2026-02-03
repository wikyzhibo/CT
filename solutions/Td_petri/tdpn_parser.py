
import math

class TDPNParser:
    def __init__(self):
        self.pn_interval = 5
    
    def parse(self, events):
        """
        Parses a list of events into a sequence of (a1, a2) transition names for PN.
        
        events: list of (time, arm, kind, from_loc, to_loc)
        Returns: list of (tm2_action_name, tm3_action_name)
        """
        # 1. Map events
        mapped_events = []
        for t, arm, kind, from_loc, to_loc in events:
            pn_action, robot = self._map_action_v2(arm, kind, from_loc, to_loc)
            if pn_action:
                mapped_events.append((t, pn_action, robot))
        
        if not mapped_events:
            return []

        # 2. Group by 5s windows
        max_time = max(e[0] for e in mapped_events)
        duration = int(math.ceil(max_time / self.pn_interval)) + 2
        
        sequence = []
        mapped_events.sort(key=lambda x: x[0])
        
        current_event_idx = 0
        n_events = len(mapped_events)
        
        for i in range(duration):
            time_start = i * self.pn_interval
            time_end = (i + 1) * self.pn_interval
            
            tm2_act = None
            tm3_act = None
            
            while current_event_idx < n_events:
                t, act, robot = mapped_events[current_event_idx]
                
                if t >= time_end:
                    break
                
                if t >= time_start or (t > time_start - 2): 
                    if robot == "TM2":
                        if tm2_act is None:
                            tm2_act = act
                        else:
                            break 
                    elif robot == "TM3":
                        if tm3_act is None:
                            tm3_act = act
                        else:
                            break
                
                current_event_idx += 1
            
            sequence.append((tm2_act, tm3_act))
            
            if current_event_idx >= n_events:
                break
        
        # Trim leading pure Wait steps
        while sequence and sequence[0] == (None, None):
            sequence.pop(0)
            
        return sequence

    def _map_action_v2(self, arm, kind, from_loc, to_loc):
        """
        Maps (arm, kind, from, to) to (PN_transition, Robot).
        kind: 0=PICK, 1=LOAD
        """
        # Normalize
        from_loc = from_loc.upper()
        if not to_loc: to_loc = "" # handle None
        to_loc = to_loc.upper()
        
        # --- Route 1 (LP1) ---
        if arm == 2:
            # LLA_S2 -> PM7/8
            if "LLA_S2" in from_loc and ("PM7" in to_loc or "PM8" in to_loc):
                if kind == 0: return "u_LP1_s1", "TM2"
                if kind == 1: return "t_s1", "TM2"
            
            # PM7/8 -> LLC (s1 -> s2)
            if ("PM7" in from_loc or "PM8" in from_loc) and "LLC" in to_loc:
                if kind == 0: return "u_s1_s2", "TM2" 
                if kind == 1: return "t_s2", "TM2"
            
            # LLD -> PM9/10 (s4 -> s5)
            if "LLD" in from_loc and ("PM9" in to_loc or "PM10" in to_loc):
                if kind == 0: return "u_s4_s5", "TM2"
                if kind == 1: return "t_s5", "TM2"
            
            # PM9/10 -> LLB (s5 -> LP_done)
            if ("PM9" in from_loc or "PM10" in from_loc) and "LLB" in to_loc:
                if kind == 0: return "u_s5_LP_done", "TM2"
                if kind == 1: return "t_LP_done", "TM2"

        if arm == 3:
            # LLC -> PM1-4 (s2 -> s3)
            if "LLC" in from_loc and "PM" in to_loc:
                if any(x in to_loc for x in ["PM1", "PM2", "PM3", "PM4"]):
                    if kind == 0: return "u_s2_s3", "TM3"
                    if kind == 1: return "t_s3", "TM3"
            
            # PM1-4 -> LLD (s3 -> s4)
            if "PM" in from_loc and "LLD" in to_loc:
                if any(x in from_loc for x in ["PM1", "PM2", "PM3", "PM4"]):
                    if kind == 0: return "u_s3_s4", "TM3"
                    if kind == 1: return "t_s4", "TM3"
                    
        return None, None
    
    def _map_action(self, t_name):
        # Deprecated legacy mapping
        return None, None
