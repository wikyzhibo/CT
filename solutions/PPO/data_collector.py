import torch
from tensordict import TensorDict


class DeadlockSafeCollector:
    """Manually collect rollouts without implicit truncation.

    The default :class:`torchrl.collectors.SyncDataCollector` will stop and
    truncate a rollout once an environment reports a terminal signal.  For
    the cluster tool deadlock cases we want to *record* the terminal transition
    but immediately reset the environment and continue collecting until
    ``frames_per_batch`` transitions are accumulated.  This lightweight
    iterator mirrors the basic output structure of ``SyncDataCollector`` while
    keeping the full trajectories, letting PPO/GAE operate on the raw data.
    """

    def __init__(
        self,
        env,
        policy,
        frames_per_batch: int,
        total_frames: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.policy = policy
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.device = torch.device(device)

        self._frames = 0
        self._pending_td = None

    def __iter__(self):
        self._frames = 0
        self._pending_td = None
        return self

    def _env_reset(self):
        self._pending_td = self.env.reset().to(self.device)

    def _is_done(self, next_td: TensorDict) -> bool:
        terminated = next_td.get("terminated")
        truncated = next_td.get("truncated")

        done = None
        if terminated is not None:
            done = terminated
        if truncated is not None:
            done = truncated if done is None else (done | truncated)

        if done is None:
            return False
        return bool(done.view(-1).any().item())

    def __next__(self) -> TensorDict:
        if self._frames >= self.total_frames:
            raise StopIteration

        steps = []
        while len(steps) < self.frames_per_batch and self._frames < self.total_frames:
            if self._pending_td is None:
                self._env_reset()

            with torch.no_grad():
                rollout_td = self._pending_td.clone()
                rollout_td = self.policy(rollout_td)
                next_td = self.env.step(rollout_td)

            step_td = rollout_td.clone()
            step_td.set("next", next_td)
            steps.append(step_td)

            self._frames += 1
            self._pending_td = None if self._is_done(next_td) else next_td.to(self.device)

        return TensorDict.stack(steps, dim=0)
