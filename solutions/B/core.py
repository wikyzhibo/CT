import numpy as np
from typing import Dict, List, Optional, Tuple
from .clustertool_config import ClusterToolCfg
from .construct import build_pdr_net
from .pn_models import FlatMarks

INF = 10**6
LEAF_NODES = []
LEAF_CLOCKS = []
LEAF_PATHS = []
LEAF_PATH_RECORDS = []

def _clear_leaf_buffers() -> None:
    global LEAF_NODES, LEAF_CLOCKS, LEAF_PATHS, LEAF_PATH_RECORDS
    LEAF_NODES = []
    LEAF_CLOCKS = []
    LEAF_PATHS = []
    LEAF_PATH_RECORDS = []

class ClusterTool:
    def __init__(self) -> None:
        cfg = ClusterToolCfg.load()
        self.n_wafer = int(cfg.n_wafer1) + int(cfg.n_wafer2)
        self.max_time = cfg.MAX_TIME
        self.done_event_reward = cfg.done_event_reward
        self.finish_event_reward = self.done_event_reward * 6
        self.scrap_event_penalty = cfg.scrap_event_penalty
        self.processing_coef_reward = cfg.processing_coef_reward
        self.time_coef_penalty = cfg.time_coef_penalty
        self.d_residual_time = cfg.D_Residual_time
        self.p_residual_time = cfg.P_Residual_time
        self.over_time = 0
        self.makespan = 0
        self.ttime = cfg.ttime
        self.search_depth = cfg.search_depth
        self.candidate_k = cfg.candidate_k
        info = build_pdr_net(n_wafer=self.n_wafer, takt_cycle=cfg.takt_cycle)
        self.id2p_name: List[str] = list(info["id2p_name"])
        self.id2t_name: List[str] = list(info["id2t_name"])
        self.P = int(len(self.id2p_name))
        self.T = int(len(self.id2t_name))
        self.m0 = info["m0"]
        self.m = self.m0.copy()
        self.ptime = np.ascontiguousarray(info["ptime"], dtype=np.int32)
        self.ttime_by_t = np.ascontiguousarray(info["ttime"], dtype=np.int32)
        self.k = np.ascontiguousarray(info["capacity"], dtype=np.int32)
        self.idle_idx = dict(info["idle_idx"])
        self.terminal_place_idx = int(self.idle_idx["end"])
        self.place_type_arr = np.ascontiguousarray(info["place_type_arr"], dtype=np.int8)
        self.place_cat = np.ascontiguousarray(info["place_cat"], dtype=np.int8)
        self._multi_wafer_pids: set[int] = {int(p) for p in info["_multi_wafer_pids"]}
        self._resource_pids: set[int] = {int(p) for p in info["_resource_pids"]}
        self._init_fm = info["marks"].clone()
        self.marks: FlatMarks = self._init_fm.clone()
        self._pre_places_idx: List[np.ndarray] = info["pre_place_cache"]
        self._pst_places_idx: List[np.ndarray] = info["pst_place_cache"]
        self._downstream_block_tids: Dict[int, np.ndarray] = {}
        downstream_block_map: Dict[str, List[str]] = dict(info.get("downstream_block_map", {}))
        p_name_to_idx = {name: idx for idx, name in enumerate(self.id2p_name)}
        for tid, t_name in enumerate(self.id2t_name):
            if not t_name.startswith("u_"):
                continue
            body = t_name[2:]
            parts = body.rsplit("_", 2)
            if len(parts) != 3:
                continue
            src_name = parts[0]
            dst_names = downstream_block_map.get(src_name, [])
            if not dst_names:
                continue
            dst_idx = [p_name_to_idx[name] for name in dst_names if name in p_name_to_idx]
            if dst_idx:
                self._downstream_block_tids[tid] = np.asarray(dst_idx, dtype=np.int32)
        self.full_transition_path: List[str] = []
        self.full_transition_records: List[Dict[str, int | str]] = []
        self.train_current_marks: Optional[FlatMarks] = None
        self._cur_clock: int = 0
        self._train_cached_candidates: Optional[Dict[str, object]] = None
        self._lp_done_idx: int = int(self.terminal_place_idx)
        self._obs_place_indices = np.asarray(
            [p for p in range(self.P) if p != self._lp_done_idx],
            dtype=np.int32,
        )
        self._obs_global_dim: int = 2 + int(self.candidate_k)
        self._obs_place_offsets = (
            self._obs_global_dim + 2 * np.arange(len(self._obs_place_indices), dtype=np.int32)
        )
        self.obs_dim: int = int(self._obs_global_dim + 2 * len(self._obs_place_indices))
        self._obs_buffer = np.zeros(self.obs_dim, dtype=np.float32)
        self._lp_place_idx: int = int(self.id2p_name.index("LP")) if "LP" in self.id2p_name else -1
        self._lp_min_release_interval: int = 180
        self._u_lp_tm2_tids: set[int] = {
            int(tid)
            for tid, t_name in enumerate(self.id2t_name)
            if t_name.startswith("u_LP_TM2_")
        }
        self._is_lp_like = np.asarray(
            [name.startswith("LP") for name in self.id2p_name],
            dtype=bool,
        )
        self._ptime_safe = np.maximum(
            self.ptime.astype(np.float32, copy=False),
            1.0,
        )
        self._max_time_safe: float = float(max(1, int(self.max_time)))
        self._delta_norm_denom: float = 200.0

    def _resident_limit_for_place(self, p: int) -> Optional[int]:
        if int(self.place_type_arr[p]) == 1:
            return int(self.p_residual_time)
        elif int(self.place_type_arr[p]) == 2:
            return int(self.d_residual_time)
        return None

    def _head_enter_time(self, p: int, fm: FlatMarks) -> int:
        cat = self.place_cat[p]
        if cat == 0:
            return int(fm.token_enter_time[fm.place_token[p]])
        elif cat == 1:
            tid = fm.wafer_queues[p][0]
            return int(fm.token_enter_time[tid])
        else:
            return int(fm.resource_queues[p][0])


    def reset(self):
        self.m = self.m0.copy()
        self.marks = self._init_fm.clone()
        self.train_current_marks = self.marks.clone()
        self._cur_clock = 0
        self._train_cached_candidates = None
        _clear_leaf_buffers()

    def get_obs(self) -> np.ndarray:
        obs = self._obs_buffer
        obs.fill(0.0)

        done_wafer = float(self.m[self.terminal_place_idx])
        obs[0] = np.float32(done_wafer / float(max(1, int(self.n_wafer))))
        obs[1] = np.float32(np.clip(float(self._cur_clock) / self._max_time_safe, 0.0, 1.0))

        delta_start = 2
        delta_end = delta_start + int(self.candidate_k)
        obs[delta_start:delta_end] = 1.0
        prepared = self._train_cached_candidates
        if prepared is not None:
            deltas = prepared.get("candidate_deltas", [])
            valid_count = min(len(deltas), int(self.candidate_k))
            if valid_count > 0:
                delta_arr = np.asarray(deltas[:valid_count], dtype=np.float32)
                delta_arr = np.clip(delta_arr / self._delta_norm_denom, 0.0, 1.0)
                obs[delta_start : delta_start + valid_count] = delta_arr

        fm = self.train_current_marks if self.train_current_marks is not None else self.marks
        cur_clock = int(self._cur_clock)
        for idx, p in enumerate(self._obs_place_indices):
            p_int = int(p)
            base = int(self._obs_place_offsets[idx])
            if int(self.place_cat[p_int]) == 2:
                continue
            if int(self.m[p_int]) <= 0:
                continue

            obs[base] = 1.0
            if self._is_lp_like[p_int]:
                continue

            if int(self.place_cat[p_int]) == 0:
                tid = int(fm.place_token[p_int])
            else:
                queue = fm.wafer_queues.get(p_int, [])
                tid = int(queue[0]) if queue else -1

            if tid >= 0:
                stay = float(cur_clock - int(fm.token_enter_time[tid])) / float(self._ptime_safe[p_int])
                obs[base + 1] = np.float32(max(0.0, stay))

        return obs.copy()

    def prepare_train_candidates(self, candidate_k: Optional[int] = None) -> Dict[str, object]:
        candidate_k = int(self.candidate_k if candidate_k is None else candidate_k)
        _clear_leaf_buffers()
        self.collect_leaves_iterative(
            m=self.m,
            fm=self.train_current_marks,
            clock=int(self._cur_clock),
            depth=self.search_depth,
        )


        if len(LEAF_NODES) == 0:
            prepared = {
                "has_candidate": False,
                "candidate_k": int(candidate_k),
                "valid_count": 0,
                "action_mask": np.zeros(int(candidate_k), dtype=bool),
                "candidate_deltas": [],
                "candidate_states": [],
            }
            self._train_cached_candidates = prepared
            return prepared

        scored: List[Tuple[int, int]] = []
        for idx in range(len(LEAF_NODES)):
            delta_clock = abs(int(LEAF_CLOCKS[idx]) - int(self._cur_clock))
            scored.append((int(idx), int(delta_clock)))
        scored.sort(key=lambda x: x[1])
        selected = scored[: int(candidate_k)]

        candidate_states: List[Dict[str, object]] = []
        candidate_deltas: List[int] = []
        for leaf_idx, delta_clock in selected:
            leaf = LEAF_NODES[int(leaf_idx)]
            candidate_states.append(
                {
                    "m": leaf["m"].copy(),
                    "marks": leaf["marks"].clone(),
                    "clock": int(LEAF_CLOCKS[int(leaf_idx)]),
                    "transition_path": list(LEAF_PATHS[int(leaf_idx)]),
                    "transition_records": [dict(item) for item in LEAF_PATH_RECORDS[int(leaf_idx)]],
                }
            )
            candidate_deltas.append(int(delta_clock))

        valid_count = len(candidate_states)
        action_mask = np.zeros(int(candidate_k), dtype=bool)
        action_mask[:valid_count] = True
        prepared = {
            "has_candidate": True,
            "candidate_k": int(candidate_k),
            "valid_count": int(valid_count),
            "action_mask": action_mask,
            "candidate_deltas": candidate_deltas,
            "candidate_states": candidate_states,
        }
        self._train_cached_candidates = prepared
        _clear_leaf_buffers()
        return prepared

    def get_enable_t(
        self,
        m: np.ndarray,
        fm: FlatMarks,
        start_from: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单次遍历所有变迁：先做 m/k 下的结构使能，再读 marks 算最早使能时刻。
        返回 (transition_ids, enable_times)，仅含可触发变迁；不构造 (T,) 掩码。
        """
        tau = start_from
        ts_list: List[int] = []
        ets_list: List[int] = []
        for t in range(self.T):
            pre_idx = self._pre_places_idx[t]
            if pre_idx.size > 0 and (m[pre_idx] < 1).any():
                continue
            pst_idx = self._pst_places_idx[t]
            if self.k is not None and pst_idx.size > 0:
                if (m[pst_idx] > self.k[pst_idx]).any():
                    continue
            blocked_dst = self._downstream_block_tids.get(t)
            if blocked_dst is not None:
                if np.all(m[blocked_dst] >= self.k[blocked_dst]):
                    continue

            earliest = tau
            for p in pre_idx:
                tok_enter = self._head_enter_time(int(p), fm) + int(self.ptime[p])
                earliest = max(earliest, tok_enter)
            ts_list.append(t)
            ets_list.append(earliest)
        if not ts_list:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        return np.asarray(ts_list, dtype=np.int32), np.asarray(ets_list, dtype=np.int32)

    def _state_key_from_m(self, m: np.ndarray, fm: FlatMarks) -> Tuple[object, ...]:
        """
        将标识与关键 marks 状态共同转换为可哈希键，用于 DFS 去重。
        """
        return (
            tuple(int(x) for x in m.tolist()),
            tuple(int(x) for x in fm.token_place.tolist()),
            tuple(int(x) for x in fm.token_enter_time.tolist()),
            tuple(int(x) for x in fm.place_token.tolist()),
            tuple((int(p), tuple(int(tid) for tid in q)) for p, q in sorted(fm.wafer_queues.items())),
            tuple((int(p), tuple(int(t) for t in q)) for p, q in sorted(fm.resource_queues.items())),
        )

    def _fire(self,
              t: int,
              m: np.ndarray,
              fm: FlatMarks,
              start_from: Optional[int] = None,
              ) -> Tuple[np.ndarray, FlatMarks, int]:

        new_m = m.copy()
        new_fm = fm.clone()

        te = start_from
        d = int(self.ttime_by_t[int(t)])
        tf = te + d - 1
        enter_new = tf + 1

        consumed_tid = -1

        for p in self._pre_places_idx[t]:
            p = int(p)
            cat = self.place_cat[p]
            if cat == 0:
                consumed_tid = int(new_fm.place_token[p])
                new_fm.place_token[p] = -1
                new_fm.token_place[consumed_tid] = -1
            elif cat == 1:
                consumed_tid = new_fm.wafer_queues[p].pop(0)
                new_fm.token_place[consumed_tid] = -1
            else:
                new_fm.resource_queues[p].pop(0)
            new_m[p] -= 1

        if int(t) in self._u_lp_tm2_tids and self._lp_place_idx >= 0:
            lp_queue = new_fm.wafer_queues.get(self._lp_place_idx, [])
            if lp_queue:
                next_head_tid = int(lp_queue[0])
                min_enter_time = int(te) + int(self._lp_min_release_interval)
                old_enter_time = int(new_fm.token_enter_time[next_head_tid])
                if old_enter_time < min_enter_time:
                    new_fm.token_enter_time[next_head_tid] = int(min_enter_time)

        for p in self._pst_places_idx[t]:
            p = int(p)
            cat = self.place_cat[p]
            if cat == 0:
                new_fm.place_token[p] = consumed_tid
                new_fm.token_place[consumed_tid] = p
                new_fm.token_enter_time[consumed_tid] = enter_new
            elif cat == 1:
                new_fm.wafer_queues[p].append(consumed_tid)
                new_fm.token_place[consumed_tid] = p
                new_fm.token_enter_time[consumed_tid] = enter_new
            else:
                new_fm.resource_queues[p].append(enter_new)
            new_m[p] += 1

        return new_m, new_fm, enter_new

    def check_scrap(self, t: int, firetime: int, fm: FlatMarks) -> bool:
        """
        检查当前动作是否触发驻留时间违规（resident scrap）。
        返回 True 表示该分支需要剪枝。
        """
        for p in range(self.P):
            resident_limit = self._resident_limit_for_place(int(p))
            if resident_limit is None:
                continue
            cat = int(self.place_cat[p])
            tid_list: List[int] = []
            if cat == 0:
                tid = int(fm.place_token[p])
                if tid >= 0:
                    tid_list.append(tid)
            elif cat == 1:
                tid_list.extend(int(tid) for tid in fm.wafer_queues.get(int(p), []))
            else:
                continue
            for tid in tid_list:
                limit_time = int(fm.token_enter_time[tid]) + int(self.ptime[p]) + int(resident_limit)
                if firetime > limit_time:
                    return True

        return False

    def calc_reward(
        self,
        delta_clock: int,
        finish: bool,
        scrap: bool,
        done_wafer_delta: int,
    ) -> float:
        reward = -float(self.time_coef_penalty) * float(delta_clock)
        if done_wafer_delta > 0:
            reward += float(done_wafer_delta) * float(self.done_event_reward)
        if finish:
            reward += float(self.finish_event_reward)
        if scrap:
            reward += float(self.scrap_event_penalty)
        return float(reward)

    def step(self, action_idx: Optional[int] = None, mode: str = "train") -> Tuple[np.ndarray, float, bool, np.ndarray, Dict[str, object]]:
        prepared = self._train_cached_candidates
        if prepared is None:
            self.reset()
            self.prepare_train_candidates()
            prepared = self._train_cached_candidates
        assert prepared is not None

        has_candidate = bool(prepared["has_candidate"])

        # 当前节点经过5层搜索没有找到任何候选叶子节点，说明当前状态不可达终止状态，直接惩罚并结束。
        if not has_candidate:
            obs = self.get_obs()
            reward = self.calc_reward(
                delta_clock=0,
                finish=False,
                scrap=True,
                done_wafer_delta=0,
            )
            next_mask = np.zeros(self.candidate_k, dtype=bool)
            done = True
            info = {
                "scrap": True,
                "delta_clock": 0,
                "finish": False,
                "candidate_count": 0,
                "time": int(self._cur_clock),
            }
            return obs, reward, done, next_mask, info

        valid_count = int(prepared["valid_count"])

        if action_idx is None:
            action_idx = int(np.random.randint(valid_count))
        else:
            action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= valid_count:
            raise IndexError(f"action_idx out of range: {action_idx}, valid_count={valid_count}")

        # 更新当前状态为选中叶子节点的状态
        prev_done_wafer = int(self.m[self.terminal_place_idx])
        selected_state = prepared["candidate_states"][action_idx]
        delta_clock = prepared["candidate_deltas"][action_idx]
        if mode == "eval":
            fired = selected_state.get("transition_path", [])
            fired_records = selected_state.get("transition_records", [])
            if fired_records:
                fired_with_time = " -> ".join(
                    f"{str(item.get('transition', ''))}@{int(item.get('fire_time', 0))}"
                    for item in fired_records
                )
                print(f"[eval] candidates:{valid_count}")
                print(f"[eval] fired transitions: {fired_with_time}")
            elif fired:
                print(f"[eval] candidates:{valid_count}")
                print(f"[eval] fired transitions: {' -> '.join(str(t) for t in fired)}")
            else:
                print("[eval] fired transitions: <none>")
        self.m = selected_state["m"].copy()
        self.train_current_marks = selected_state["marks"].clone()
        self._cur_clock = int(selected_state["clock"])

        cur_done_wafer = int(self.m[self.terminal_place_idx])
        done_wafer_delta = max(0, cur_done_wafer - prev_done_wafer)
        finish = bool(cur_done_wafer == self.n_wafer)


        if finish:
            next_mask = np.zeros(self.candidate_k, dtype=bool)
            next_has_candidate = False
        else:
            next_prepared = self.prepare_train_candidates()
            next_mask = np.asarray(next_prepared["action_mask"], dtype=bool)
            next_has_candidate = bool(next_prepared["has_candidate"])

        if (not finish) and self._cur_clock >= self.max_time:
            next_mask = np.zeros(self.candidate_k, dtype=bool)
            next_has_candidate = False

        obs = self.get_obs()
        scrap = (not bool(finish)) and (not bool(next_has_candidate))
        reward = self.calc_reward(
            delta_clock=int(delta_clock),
            finish=bool(finish),
            scrap=bool(scrap),
            done_wafer_delta=int(done_wafer_delta),
        )
        info = {
            "scrap": scrap,
            "delta_clock": int(delta_clock),
            "finish": bool(finish),
            "candidate_count": int(valid_count),
            "time": int(self._cur_clock),
        }
        done = bool(finish) or bool(scrap)
        return obs, reward, done, next_mask, info

    def get_leaf_node(
            self,
            m: np.ndarray,
            fm: FlatMarks,
            clock: int,
            path: List[int],
            path_records: List[Dict[str, int | str]],
    ) -> None:
        """
        记录深度叶子节点（全局收集）。
        """
        global LEAF_NODES, LEAF_CLOCKS, LEAF_PATHS, LEAF_PATH_RECORDS
        LEAF_NODES.append({
            "m": m.copy(),
            "marks": fm.clone(),
        })
        LEAF_CLOCKS.append(int(clock))
        LEAF_PATHS.append([self.id2t_name[t] for t in path])
        LEAF_PATH_RECORDS.append([{"transition": str(item["transition"]), "fire_time": int(item["fire_time"])} for item in path_records])

    def collect_leaves_iterative(
            self,
            m: np.ndarray,
            fm: FlatMarks,
            clock: int,
            depth: int,
    ) -> None:
        """
        使用显式栈执行 DFS，避免递归。
        栈帧: (m, fm, clock, depth, path, path_records)
        """
        stack = [(m.copy(), fm.clone(), int(clock), int(depth), [], [])]
        seen: set[Tuple[object, ...]] = {self._state_key_from_m(m, fm)}

        while stack:
            cur_m, cur_fm, cur_clock, cur_depth, cur_path, cur_path_records = stack.pop()

            if bool(int(cur_m[self.terminal_place_idx]) == self.n_wafer) or cur_depth == 0:
                self.get_leaf_node(
                    m=cur_m,
                    fm=cur_fm,
                    clock=cur_clock,
                    path=cur_path,
                    path_records=cur_path_records,
                )
                continue

            ts, ets = self.get_enable_t(
                cur_m,
                cur_fm,
                cur_clock,
            )
            if len(ts) == 0:
                continue

            transition_queue = [
                (int(t), int(et), self.id2t_name[int(t)]) for t, et in zip(ts, ets)
            ]

            for t, enable_time, _ in reversed(transition_queue):
                if self.check_scrap(t, enable_time, cur_fm):
                    self.over_time += 1
                    continue

                new_m, new_fm, new_clock = self._fire(t, cur_m, cur_fm, start_from=enable_time)

                new_path = cur_path + [int(t)]
                new_path_records = cur_path_records + [{"transition": self.id2t_name[t], "fire_time": int(enable_time)}]
                state_key = self._state_key_from_m(new_m, new_fm)
                if state_key in seen:
                    continue
                seen.add(state_key)
                stack.append((new_m, new_fm, new_clock, cur_depth - 1, new_path, new_path_records))
