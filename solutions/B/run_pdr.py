import argparse
import cProfile
import io
import pstats
import time
import random

from . import core as core_module
from .core import ClusterTool, _clear_leaf_buffers
from .parse_sequences import export_single_replay_payload


def select_node(queue, current_clock: int, mode=0):
    """根据SPT原则在候选叶子节点中选择一个节点"""
    scored = []

    match mode:
        case 0:
            for leaf_idx in queue:
                delta_clock = abs(int(core_module.LEAF_CLOCKS[int(leaf_idx)]) - int(current_clock))
                scored.append((int(leaf_idx), int(delta_clock)))
            scored.sort(key=lambda x: x[1])
            return scored[0][0]
        case 1:
            return random.randint(0, len(queue) - 1)



def search(net: ClusterTool) -> bool:
    """
    迭代 DFS：
    1) 以当前状态做深度受限 DFS，收集叶子
    2) 选择一个叶子作为当前状态
    3) 清空叶子集合，继续下一轮 DFS
    4) 直到到达终止状态
    """
    net.full_transition_path = []
    net.full_transition_records = []
    current_m = net.m.copy()
    current_fm = net.marks.clone()
    current_clock = 0

    max_round = 1000
    round_idx = 0
    while True:
        if round_idx >= max_round:
            print(f"Reached max round {max_round}, terminating search.")
            break
        round_idx += 1

        if bool(int(current_m[net.terminal_place_idx]) == net.n_wafer):
            net.m = current_m.copy()
            net.marks = current_fm.clone()
            net.time = int(current_clock)
            net.makespan = int(current_clock)
            return True

        _clear_leaf_buffers()
        net.transitions = []
        net.time_record = []
        net.collect_leaves_iterative(
            m=current_m,
            fm=current_fm,
            clock=int(current_clock),
            depth=net.search_depth,
        )
        if len(core_module.LEAF_NODES) == 0:
            print("[WARN] no DFS leaf node can be expanded, terminating search.")
            break

        candidate_indices = list(range(len(core_module.LEAF_NODES)))
        leaf_idx = select_node(candidate_indices, current_clock=current_clock, mode=1)
        net.full_transition_path.extend(core_module.LEAF_PATHS[leaf_idx])
        net.full_transition_records.extend(core_module.LEAF_PATH_RECORDS[leaf_idx])

        current_m = core_module.LEAF_NODES[leaf_idx]["m"].copy()
        current_fm = core_module.LEAF_NODES[leaf_idx]["marks"].clone()
        current_clock = int(core_module.LEAF_CLOCKS[leaf_idx])

        _clear_leaf_buffers()

    return False


def main():
    start = time.time()
    net = ClusterTool()
    net.reset()
    ok = search(net)
    print(net.full_transition_path)
    if ok:
        replay_path = export_single_replay_payload(net.full_transition_records, out_name="pdr_sequence")
        print(f"[INFO] replay sequence exported: {replay_path} | records={len(net.full_transition_records)}")
        print(f'makespan={net.makespan}|search time = {time.time()-start:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDR Petri search / replay export")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run main() under cProfile and print top functions by cumulative time",
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default="",
        metavar="PATH",
        help="Write raw profile to PATH (e.g. for python -m pstats or snakeviz)",
    )
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=50,
        metavar="N",
        help="Number of lines to print when using --profile (default: 50)",
    )
    args = parser.parse_args()

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        try:
            main()
        finally:
            prof.disable()
            if args.profile_out:
                prof.dump_stats(args.profile_out)
                print(f"[PROFILE] wrote {args.profile_out}")
            buf = io.StringIO()
            stats = pstats.Stats(prof, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
            stats.print_stats(int(args.profile_lines))
            print(buf.getvalue())
    else:
        main()
