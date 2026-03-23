import time
from solutions.PDR.net import Petri
from solutions.PDR.parse_sequences import export_single_replay_payload

def main():
    start = time.time()
    net = Petri(n_wafer=7, ttime=5)
    net.reset()
    ok = net.search()
    print(net.full_transition_path)
    if ok:
        replay_path = export_single_replay_payload(net.full_transition_records, out_name="pdr_sequence")
        print(f"[INFO] replay sequence exported: {replay_path} | records={len(net.full_transition_records)}")
    print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}')


if __name__ == '__main__':
    main()