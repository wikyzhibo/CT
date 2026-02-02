
import sys
import os
sys.path.append(os.getcwd())

from solutions.Continuous_model.pn import Petri
from visualization.petri_adapter import PetriAdapter
from solutions.Continuous_model.construct import BasedToken

def test_dual_transport():
    print("=== Verification Test: Dual Transport Places (TM2/TM3) ===")
    
    # 1. Initialize Petri Net
    try:
        env = Petri()
        print("[SUCCESS] Petri Net Initialized")
    except Exception as e:
        print(f"[FAIL] Petri Net Init: {e}")
        return

    # 2. Check d_TM2, d_TM3 existence & capacity
    places = env.id2p_name
    
    if "d_TM2" in places:
        d2_idx = env._get_place_index("d_TM2")
        cap = env.marks[d2_idx].capacity
        print(f"[SUCCESS] d_TM2 found. Capacity: {cap}")
        if cap != 1:
            print("[WARN] d_TM2 capacity should be 1")
    else:
        print(f"[FAIL] d_TM2 NOT found.")

    if "d_TM3" in places:
        d3_idx = env._get_place_index("d_TM3")
        cap = env.marks[d3_idx].capacity
        print(f"[SUCCESS] d_TM3 found. Capacity: {cap}")
        if cap != 1:
            print("[WARN] d_TM3 capacity should be 1")
    else:
        print(f"[FAIL] d_TM3 NOT found.")

    # 3. Check d_transport (Should NOT exist)
    if "d_transport" in places:
        print("[FAIL] d_transport still exists!")
    else:
        print("[SUCCESS] d_transport correctly removed.")

    # 4. Check PetriAdapter mapping
    try:
        adapter = PetriAdapter(env)
        print("[SUCCESS] PetriAdapter Initialized")
    except Exception as e:
        print(f"[FAIL] PetriAdapter Init: {e}")
        return

    # 5. Check adapter transports list
    expected_trans = ["d_TM2", "d_TM3"]
    # minimal check, order might vary
    if set(adapter.transports) == set(expected_trans):
        print("[SUCCESS] Adapter transports list correct")
    else:
        print(f"[FAIL] Adapter transports list incorrect: {adapter.transports}")

    # 6. Check Robot State Logic
    print("Testing Robot State Mapping...")
    
    # Case A: Token in d_TM2
    d2_idx = env._get_place_index("d_TM2")
    tok_tm2 = BasedToken(enter_time=0, token_id=101, route_type=1, step=0)
    env.marks[d2_idx].append(tok_tm2)
    
    state = adapter.get_current_state()
    tm2_wafers = state.robot_states['TM2'].wafers
    tm3_wafers = state.robot_states['TM3'].wafers
    
    print(f"  d_TM2 injected. TM2 wafers: {len(tm2_wafers)}, TM3 wafers: {len(tm3_wafers)}")
    if len(tm2_wafers) == 1 and len(tm3_wafers) == 0:
        print("  [SUCCESS] d_TM2 mapped to TM2")
    else:
        print("  [FAIL] Mapping Incorrect")
    
    env.marks[d2_idx].pop_head() # Clean up

    # Case B: Token in d_TM3
    d3_idx = env._get_place_index("d_TM3")
    tok_tm3 = BasedToken(enter_time=0, token_id=102, route_type=1, step=2)
    env.marks[d3_idx].append(tok_tm3)
    
    state = adapter.get_current_state()
    tm2_wafers = state.robot_states['TM2'].wafers
    tm3_wafers = state.robot_states['TM3'].wafers
    
    print(f"  d_TM3 injected. TM2 wafers: {len(tm2_wafers)}, TM3 wafers: {len(tm3_wafers)}")
    if len(tm2_wafers) == 0 and len(tm3_wafers) == 1:
        print("  [SUCCESS] d_TM3 mapped to TM3")
    else:
        print("  [FAIL] Mapping Incorrect")

    print("=== End Verification ===")

if __name__ == "__main__":
    try:
        test_dual_transport()
    except Exception:
        import traceback
        traceback.print_exc()
