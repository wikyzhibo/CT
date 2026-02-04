
import sys
import os
sys.path.append(os.getcwd())

from solutions.Continuous_model.pn import Petri
from solutions.Continuous_model.construct import BasedToken

def test_violation_counters():
    print("Testing violation counters...")
    
    # 1. Initialize Petri net
    pn = Petri(enable_statistics=True)
    pn.reset()
    print("Petri net reset.")
    
    # Verify initial counters
    stats = pn.calc_wafer_statistics()
    assert stats["resident_violation_count"] == 0, "Initial resident violation count should be 0"
    assert stats["qtime_violation_count"] == 0, "Initial qtime violation count should be 0"
    
    # 2. Simulate Q-time violation (Type 2 place)
    # Find a transport place (Type 2)
    transport_places = [p for p in pn.marks if p.type == 2]
    if not transport_places:
        print("Error: No transport places found.")
        return
    
    target_p = transport_places[0]
    print(f"Injecting token into Type 2 place: {target_p.name}")
    
    # Create a token that entered long ago
    # Current time = 1
    # Q-time limit = 15s
    # Make enter_time = -20 (so stick time = 1 - (-20) = 21s > 15s)
    token = BasedToken(token_id=999, route_type=1, enter_time=-20)
    target_p.tokens.append(token)
    
    # Call _check_scrap to limit-check (it returns info)
    is_scrap, info = pn._check_scrap(return_info=True)
    
    if is_scrap:
        print(f"Scrap detected: {info}")
        if info['type'] == 'qtime':
            print("Confirmed: Detected Q-time violation.")
        else:
            print(f"Error: Detected wrong type: {info['type']}")
    else:
        print("Error: Failed to detect Q-time violation.")
        
    # Now call step() to trigger the counter increment (assuming step calls _check_scrap)
    # We must ensure stop_on_scrap doesn't exit immediately if we want to check counters,
    # or we catch the result. step() returns (done, scrap) or similar.
    # But pn.step checks scrap inside.
    
    # Force step to invoke check
    done, scrap = pn.step(wait=True)
    print(f"Step result: done={done}, scrap={scrap}")
    
    # Check counters
    stats = pn.calc_wafer_statistics()
    print(f"Stats: Res={stats['resident_violation_count']}, Qtime={stats['qtime_violation_count']}")
    
    assert stats["qtime_violation_count"] == 1, f"Expected 1 Q-time violation, got {stats['qtime_violation_count']}"
    
    # 3. Simulate Resident Time violation (Type 1 place)
    # Reset for clarity
    pn.reset()
    pn.resident_violation_count = 0 
    pn.qtime_violation_count = 0
    
    proc_places = [p for p in pn.marks if p.type == 1]
    if not proc_places:
        print("Error: No process places found.")
        return
        
    target_p = proc_places[0]
    print(f"Injecting token into Type 1 place: {target_p.name} (proc_time={target_p.processing_time})")
    
    # Limit = proc_time + P_Residual_time
    # P_Residual_time default is likely 15 or similar from config. 
    # Let's say we set enter_time such that stay_time > proc_time + P_Residual + 10
    limit = target_p.processing_time + pn.P_Residual_time
    enter_t = - (limit + 10)
    token = BasedToken(token_id=888, route_type=1, enter_time=enter_t)
    target_p.tokens.append(token)
    
    done, scrap = pn.step(wait=True)
    stats = pn.calc_wafer_statistics()
    print(f"Stats: Res={stats['resident_violation_count']}, Qtime={stats['qtime_violation_count']}")
    
    assert stats["resident_violation_count"] == 1, f"Expected 1 Resident violation, got {stats['resident_violation_count']}"
    
    print("\nSUCCESS: All violation logic verified.")

if __name__ == "__main__":
    test_violation_counters()
