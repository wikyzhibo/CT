
import sys
import os
sys.path.append(os.getcwd())

from solutions.Continuous_model.pn import Petri
from visualization.petri_adapter import PetriAdapter

def test_unified_transport():
    print("=== Verification Test: Unified Transport Place ===")
    
    # 1. Initialize Petri Net
    try:
        env = Petri()
        print("[SUCCESS] Petri Net Initialized")
    except Exception as e:
        print(f"[FAIL] Petri Net Init: {e}")
        return

    # 2. Check d_transport existence
    if "d_transport" in env.id2p_name:
        print("[SUCCESS] d_transport found in places")
    else:
        print(f"[FAIL] d_transport NOT found. Places: {env.id2p_name}")
        return

    # 3. Check d_transport capacity (Should be INF for non-blocking)
    d_idx = env._get_place_index("d_transport")
    capacity = env.marks[d_idx].capacity
    print(f"[INFO] d_transport capacity: {capacity}")

    # 4. Check PetriAdapter mapping
    try:
        adapter = PetriAdapter(env)
        print("[SUCCESS] PetriAdapter Initialized")
    except Exception as e:
        print(f"[FAIL] PetriAdapter Init: {e}")
        return

    # 5. Check adapter transports list
    if adapter.transports == ["d_transport"]:
        print("[SUCCESS] Adapter transports list correct")
    else:
        print(f"[FAIL] Adapter transports list incorrect: {adapter.transports}")

    # 6. Check Robot State Logic (Simulate token)
    print("Testing Robot State Mapping...")
    # Manually inject a token into d_transport
    from solutions.Continuous_model.construct import BasedToken
    # Route 1, Step 1 -> Next target s2 (TM2) (Config: Route1=["s1", "s2", ...])
    # Route1: 0=LP, 1=s1, 2=s2... NO.
    # Route Config: 1: ["s1", "s2", "s3", "s4", "s5", "LP_done"]
    # step=0 -> s1 (TM2)
    # step=1 -> s2 (TM3? No, s2 is LLC -> TM3?)
    # Config: s2 -> TM3?
    # Adapter config: s2 source -> LLC. LLC robot="TM3".
    
    # Case A: Token going to s1 (TM2)
    # route=1, step=0. Target=s1.
    tok_tm2 = BasedToken(enter_time=0, token_id=1, route_type=1, step=0)
    env.marks[d_idx].append(tok_tm2)
    
    state = adapter.get_current_state()
    tm2_wafers = state.robot_states['TM2'].wafers
    tm3_wafers = state.robot_states['TM3'].wafers
    print(f"  Case A (Target s1/TM2): TM2={len(tm2_wafers)}, TM3={len(tm3_wafers)}")
    if len(tm2_wafers) == 1 and len(tm3_wafers) == 0:
        print("  [SUCCESS] Mapped to TM2")
    else:
        print("  [FAIL] Mapping Incorrect")

    env.marks[d_idx].pop_head() # Clean up

    # Case B: Token going to s2 (TM3)
    # route=1, step=1. Target=s2. s2 robot=?
    # Adapter config: "LLC": {"source": "s2", "robot": "TM3"}
    # Wait, source=s2 means wafer is AT s2.
    # But d_transport contains wafer GOING TO s2?
    # Logic in adapter: target = self.net._get_next_target(). 
    # If target is s2.
    # Adapter looks up `self.chamber_config[target]`.
    # config["s2"]?
    # Adapter config has keys: LLA, LLB, PM7, PM8, PM9, PM10, LLC, LLD, PM1-4.
    # It does NOT have "s1", "s2" as keys (except implicitly via map?).
    # let's check adapter.chamber_config keys.
    # Lines 36-51: Keys are "LLA", "LLB", "PM7"...
    # But `_get_next_target` returns "s1", "s2", "s5".
    # Does adapter config have entries for "s1", "s2"?
    # NO. It has keys like "PM7" with source="s1".
    
    # CRITICAL: If _get_next_target returns "s1", and we look up "s1" in chamber_config...
    # We will FAIL if chamber_config doesn't have "s1".
    # I need to check how I implemented `_collect_robot_states`!
    
    # Implementation:
    # if target and target in self.chamber_config:
    #     robot = self.chamber_config[target].get("robot")
    
    # If target is "s1", "s1" is NOT in chamber_config keys.
    # chamber_config keys are CHAMBER NAMES (PM7, LLC...).
    # s1, s2 are PLACE NAMES.
    
    # So `_collect_robot_states` implementation is BUGGY if it expects place names as keys.
    # I need to map Place Name -> Chamber Name -> Robot.
    # Or Place Name -> Robot.
    
    print(f"  [DEBUG] Config keys sample: {list(adapter.chamber_config.keys())[:3]}")
    # We need to find which chamber has source=target.
    
    print("=== End Verification ===")

if __name__ == "__main__":
    test_unified_transport()
