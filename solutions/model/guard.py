import heapq
from typing import Dict, Optional, Any, Tuple


class ZeroBufferWindowController:
    """
    三工序(1->2->3)：
    - commit t3(stage1)：预约 S2 的机器时间段，并记录到 job_id
    - commit t5(stage2)：用真实开工时间回写 S2 预约（允许比预约晚，会推迟机器释放） + 预约 S3
    - commit t7(stage3)：用真实开工时间回写 S3 预约（允许比预约晚，会推迟机器释放）

    约定：
    - window[1] = w12：S1 完成+tau 到达 S2 后，需在 w12 内开工（用于 peek 反推）
    - window[2] = w23：S2 完成+tau 到达 S3 后，需在 w23 内开工（用于 peek 反推）
    """

    def __init__(
        self,
        window: Dict[int, float],       # {1:w12, 2:w23}
        proc_time: Dict[int, float],    # {1:p1, 2:p2, 3:p3}
        capacity: Dict[int, int],       # {1:c1, 2:c2, 3:c3}
        tau: float = 5.0,               # 运输时间
        t_start_stage: Optional[Dict[str, int]] = None,
    ):
        self.P = proc_time
        self.cap = capacity
        self.window = window
        self.tau = tau
        self.t_start_stage = t_start_stage or {"t3": 1, "t5": 2, "t7": 3}

        # 工序占用（真实 commit）：min-heap of finish times
        self.busy_finish_heap = {1: [], 2: [], 3: []}

        # S2/S3 预约：机器可用时刻堆 (avail_time, machine_id)
        self.accept_heap_2 = [(0.0, mid) for mid in range(self.cap[2])]
        heapq.heapify(self.accept_heap_2)

        self.accept_heap_3 = [(0.0, mid) for mid in range(self.cap[3])]
        heapq.heapify(self.accept_heap_3)

        # job_id -> reservation
        # resv2[job] = (mid2, reserved_s2_start, reserved_s2_finish)
        # resv3[job] = (mid3, reserved_s3_start, reserved_s3_finish)
        self.resv2: Dict[Any, Tuple[int, float, float]] = {}
        self.resv3: Dict[Any, Tuple[int, float, float]] = {}

    def reset(self):
        self.busy_finish_heap = {1: [], 2: [], 3: []}

        self.accept_heap_2 = [(0.0, mid) for mid in range(self.cap[2])]
        heapq.heapify(self.accept_heap_2)

        self.accept_heap_3 = [(0.0, mid) for mid in range(self.cap[3])]
        heapq.heapify(self.accept_heap_3)

        self.resv2 = {}
        self.resv3 = {}

    # ---------- DFS 必备：快照/恢复 ----------
    def snapshot(self):
        return (
            {k: list(v) for k, v in self.busy_finish_heap.items()},
            list(self.accept_heap_2),
            list(self.accept_heap_3),
            dict(self.resv2),
            dict(self.resv3),
        )

    def restore(self, snap):
        busy, acc2, acc3, resv2, resv3 = snap

        self.busy_finish_heap = {k: list(v) for k, v in busy.items()}
        heapq.heapify(self.busy_finish_heap[1])
        heapq.heapify(self.busy_finish_heap[2])
        heapq.heapify(self.busy_finish_heap[3])

        self.accept_heap_2 = list(acc2)
        heapq.heapify(self.accept_heap_2)

        self.accept_heap_3 = list(acc3)
        heapq.heapify(self.accept_heap_3)

        self.resv2 = dict(resv2)
        self.resv3 = dict(resv3)

    # ---------- 内部 ----------
    def _release_done(self, stage: int, now: float):
        heap = self.busy_finish_heap[stage]
        while heap and heap[0] <= now:
            heapq.heappop(heap)

    def _stage_available_time(self, stage: int, earliest: float) -> float:
        self._release_done(stage, earliest)
        heap = self.busy_finish_heap[stage]
        if len(heap) < self.cap[stage]:
            return earliest
        return heap[0]

    # ---------- 外部调用：peek ----------
    def peek_time(self, t_name: str, earliest: float) -> float:
        """
        earliest:
          - 对 stage1(t3)：是 S1 最早可开工时刻
          - 对 stage2(t5)：是 S2 最早可开工时刻（可认为已包含 tau）
          - 对 stage3(t7)：是 S3 最早可开工时刻（可认为已包含 tau）
        """
        if t_name not in self.t_start_stage:
            return earliest

        stage = self.t_start_stage[t_name]
        s_cap = self._stage_available_time(stage, earliest)

        # stage3：直接返回可用时间即可
        if stage == 3:
            return s_cap

        # stage2：用 S3 的最早预约槽反推 S2 最早可开工
        if stage == 2:
            slot3, _ = self.accept_heap_3[0]
            d_ready = max(s_cap, slot3)
            # 如果现在开 S2，finish2+tau 到 S3；要在 window[2] 内能开工
            lb = d_ready - (self.P[2] + self.tau + self.window.get(2, 0.0))
            return max(s_cap, lb)

        # stage1：用 S2 的最早预约槽反推 S1 最早可开工
        slot2, _ = self.accept_heap_2[0]
        d_ready = max(s_cap, slot2)
        # 如果现在开 S1，finish1+tau 到 S2；要在 window[1] 内能开工
        lb = d_ready - (self.P[1] + self.tau + self.window.get(1, 0.0))
        return max(s_cap, lb)

    # ---------- 外部调用：commit ----------
    def commit(self, t_name: str, start_time: float, job_id: Any = None):
        """
        - commit stage1(t3): 必须传 job_id，用于写 S2 reservation
        - commit stage2(t5): 必须传 job_id，用于回写 S2 reservation + 写 S3 reservation
        - commit stage3(t7): 必须传 job_id，用于回写 S3 reservation
        """
        if t_name not in self.t_start_stage:
            return

        stage = self.t_start_stage[t_name]

        # 基于本工序占用，修正 start_time
        start_time = max(start_time, self._stage_available_time(stage, start_time))

        # 不允许早于对应预约开始（保持 peek 推导成立）
        if stage == 2 and job_id is not None and job_id in self.resv2:
            _mid2, reserved_s2_start, _ = self.resv2[job_id]
            start_time = max(start_time, reserved_s2_start)

        if stage == 3 and job_id is not None and job_id in self.resv3:
            _mid3, reserved_s3_start, _ = self.resv3[job_id]
            start_time = max(start_time, reserved_s3_start)

        finish_time = start_time + self.P[stage]

        # 占用本工序
        self._release_done(stage, start_time)
        heap = self.busy_finish_heap[stage]
        if len(heap) < self.cap[stage]:
            heapq.heappush(heap, finish_time)
        else:
            heapq.heapreplace(heap, finish_time)

        # ---- stage1：写 S2 预约 ----
        if stage == 1:
            if job_id is None:
                raise ValueError("commit(stage1/t3) 必须传 job_id 才能做 S2 预约")

            slot2, mid2 = heapq.heappop(self.accept_heap_2)

            arrive2 = finish_time + self.tau
            reserved_s2_start = max(slot2, arrive2)
            reserved_s2_finish = reserved_s2_start + self.P[2]

            heapq.heappush(self.accept_heap_2, (reserved_s2_finish, mid2))
            self.resv2[job_id] = (mid2, reserved_s2_start, reserved_s2_finish)
            return

        # ---- stage2：回写 S2 + 写 S3 预约 ----
        if stage == 2:
            if job_id is None:
                raise ValueError("commit(stage2/t5) 必须传 job_id 才能回写 S2 / 预约 S3")

            # 先回写 S2（如果存在预约）
            if job_id in self.resv2:
                mid2, reserved_s2_start, _ = self.resv2[job_id]
                actual_s2_start = max(start_time, reserved_s2_start)
                actual_s2_finish = actual_s2_start + self.P[2]

                # 推迟 accept_heap_2[mid2]
                for i, (t_avail, m) in enumerate(self.accept_heap_2):
                    if m == mid2:
                        self.accept_heap_2[i] = (max(t_avail, actual_s2_finish), m)
                        heapq.heapify(self.accept_heap_2)
                        break

                self.resv2[job_id] = (mid2, actual_s2_start, actual_s2_finish)
            else:
                # 没走预约路径：用实际占用时间当作到达3的基准
                actual_s2_start = start_time
                actual_s2_finish = actual_s2_start + self.P[2]

            # 再预约 S3
            slot3, mid3 = heapq.heappop(self.accept_heap_3)
            arrive3 = actual_s2_finish + self.tau
            reserved_s3_start = max(slot3, arrive3)
            reserved_s3_finish = reserved_s3_start + self.P[3]

            heapq.heappush(self.accept_heap_3, (reserved_s3_finish, mid3))
            self.resv3[job_id] = (mid3, reserved_s3_start, reserved_s3_finish)
            return

        # ---- stage3：回写 S3 ----
        if stage == 3:
            if job_id is None:
                raise ValueError("commit(stage3/t7) 必须传 job_id 才能回写 S3")
            if job_id not in self.resv3:
                return

            mid3, reserved_s3_start, _ = self.resv3[job_id]
            actual_s3_start = max(start_time, reserved_s3_start)
            actual_s3_finish = actual_s3_start + self.P[3]

            for i, (t_avail, m) in enumerate(self.accept_heap_3):
                if m == mid3:
                    self.accept_heap_3[i] = (max(t_avail, actual_s3_finish), m)
                    heapq.heapify(self.accept_heap_3)
                    break

            self.resv3[job_id] = (mid3, actual_s3_start, actual_s3_finish)
            return

