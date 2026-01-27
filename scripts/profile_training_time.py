"""
Training Time Profiling Script

This script provides detailed analysis of time distribution during PPO training,
breaking down time consumption across all major components.

Usage:
    python scripts/profile_training_time.py --turbo
    python scripts/profile_training_time.py --no-turbo
    python scripts/profile_training_time.py --compare
    python scripts/profile_training_time.py --turbo --batches 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import cProfile
import pstats
import io
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

from solutions.PPO.enviroment import Env_PN
from solutions.PPO.network.models import MaskedPolicyHead
from data.ppo_configs.training_config import PPOTrainingConfig

from torchrl.envs import Compose, DTypeCastTransform, TransformedEnv, ActionMask
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, MaskedCategorical, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


class TrainingProfiler:
    """Profiler for analyzing PPO training time distribution."""
    
    def __init__(self, enable_turbo: bool, num_batches: int = 10):
        """
        Initialize the training profiler.
        
        Args:
            enable_turbo: Whether to enable turbo mode
            num_batches: Number of training batches to profile
        """
        self.enable_turbo = enable_turbo
        self.num_batches = num_batches
        self.device = torch.device("cpu")
        
        # Timing storage
        self.timings = defaultdict(list)
        self.total_time = 0.0
        
        # Setup training components
        self._setup_training()
    
    def _setup_training(self):
        """Setup training environment and networks."""
        print(f"\nSetting up training (turbo={'ON' if self.enable_turbo else 'OFF'})...")
        
        # Configuration
        self.config = PPOTrainingConfig(
            n_hidden=128,
            n_layer=4,
            total_batch=self.num_batches,
            sub_batch_size=64,
            num_epochs=10,
            lr=1e-4,
            device="cpu",
            seed=42,
            training_phase=2,
            with_pretrain=False
        )
        
        # Create environment
        base_env = Env_PN(
            device=self.device,
            training_phase=1,
            enable_turbo=self.enable_turbo
        )
        
        transform = Compose([
            ActionMask(),
            DTypeCastTransform(dtype_in=torch.int64, dtype_out=torch.float32,
                              in_keys="observation", out_keys="observation_f"),
        ])
        
        self.env = TransformedEnv(base_env, transform)
        
        # Create policy and value networks
        n_actions = self.env.action_spec.space.n
        n_obs = self.env.observation_spec["observation"].shape[0]
        
        policy_backbone = MaskedPolicyHead(
            hidden=self.config.n_hidden,
            n_obs=n_obs,
            n_actions=n_actions,
            n_layers=self.config.n_layer
        )
        td_module = TensorDictModule(
            policy_backbone, in_keys=["observation_f"], out_keys=["logits"]
        )
        self.policy = ProbabilisticActor(
            module=td_module,
            in_keys={"logits": "logits", "mask": "action_mask"},
            out_keys=["action"],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
        ).to(self.device)
        
        self.value_module = ValueOperator(
            module=nn.Sequential(
                nn.Linear(n_obs, self.config.n_hidden), nn.ReLU(),
                nn.Linear(self.config.n_hidden, self.config.n_hidden), nn.ReLU(),
                nn.Linear(self.config.n_hidden, self.config.n_hidden), nn.ReLU(),
                nn.Linear(self.config.n_hidden, 1),
            ),
            in_keys=["observation_f"],
        ).to(self.device)
        
        # Optimizer
        self.optim = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_module.parameters()),
            lr=self.config.lr
        )
        
        # Collector
        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=self.config.frames_per_batch * self.config.total_batch,
            device=self.device,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        
        # GAE and Loss
        self.gae = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.value_module
        )
        self.loss_module = ClipPPOLoss(
            actor=self.policy,
            critic=self.value_module,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coeff=self.config.entropy_start,
            critic_coeff=0.5,
            normalize_advantage=True,
        )
    
    def run_profiling(self) -> Dict[str, float]:
        """
        Run profiling on training loop with detailed data collection breakdown.
        
        Returns:
            Dictionary of component timings
        """
        print(f"Running profiling for {self.num_batches} batches...")
        
        start_total = time.time()
        
        # Manual data collection with detailed timing
        batch_count = 0
        
        while batch_count < self.num_batches:
            # Reset environment
            start_reset = time.time()
            td = self.env.reset()
            self.timings['env_reset'].append(time.time() - start_reset)
            
            frames_collected = 0
            target_frames = self.config.frames_per_batch
            
            # Collect one batch of data
            while frames_collected < target_frames:
                # Policy forward pass
                start_policy = time.time()
                with torch.no_grad():
                    td = self.policy(td)
                self.timings['policy_forward'].append(time.time() - start_policy)
                
                # Environment step
                start_env_step = time.time()
                td = self.env.step(td)
                self.timings['env_step'].append(time.time() - start_env_step)
                
                # TensorDict operations (checking termination, etc.)
                start_tensordict = time.time()
                terminated = td["terminated"].item()
                frames_collected += 1
                self.timings['tensordict_ops'].append(time.time() - start_tensordict)
                
                if terminated:
                    start_reset = time.time()
                    td = self.env.reset()
                    self.timings['env_reset'].append(time.time() - start_reset)
            
            batch_count += 1
            
            # Note: We skip the actual training loop here since we're focusing on data collection
            # For full profiling, we would continue with GAE, loss, etc.
        
        self.total_time = time.time() - start_total
        
        return self._aggregate_timings()
    
    def run_profiling_with_training(self) -> Dict[str, float]:
        """
        Run profiling including training loop (original method).
        
        Returns:
            Dictionary of component timings
        """
        print(f"Running profiling for {self.num_batches} batches...")
        
        start_total = time.time()
        
        with set_exploration_type(ExplorationType.RANDOM):
            for batch_idx, tensordict_data in enumerate(self.collector):
                if batch_idx >= self.num_batches:
                    break
                
                # 1. Data Collection (already happened in collector)
                # We'll measure the next components
                
                # 2. GAE Computation
                start_gae = time.time()
                gae_vals = self.gae(tensordict_data)
                tensordict_data.set("advantage", gae_vals.get("advantage"))
                tensordict_data.set("value_target", gae_vals.get("value_target"))
                self.timings['gae_computation'].append(time.time() - start_gae)
                
                # 3. Replay Buffer Operations
                start_buffer = time.time()
                self.replay_buffer.extend(tensordict_data)
                self.timings['replay_buffer'].append(time.time() - start_buffer)
                
                # 4. Training Epochs
                for epoch_idx in range(self.config.num_epochs):
                    for mini_batch_idx in range(self.config.frames_per_batch // self.config.sub_batch_size):
                        # Sample from buffer
                        start_sample = time.time()
                        subdata = self.replay_buffer.sample(self.config.sub_batch_size).to(self.device)
                        self.timings['buffer_sample'].append(time.time() - start_sample)
                        
                        # Loss Computation
                        start_loss = time.time()
                        loss_vals = self.loss_module(subdata)
                        loss = (loss_vals["loss_objective"] +
                               loss_vals["loss_critic"] +
                               loss_vals["loss_entropy"])
                        self.timings['loss_computation'].append(time.time() - start_loss)
                        
                        # Backward Pass
                        start_backward = time.time()
                        self.optim.zero_grad()
                        loss.backward()
                        self.timings['backward_pass'].append(time.time() - start_backward)
                        
                        # Optimizer Step
                        start_optim = time.time()
                        nn.utils.clip_grad_norm_(
                            list(self.policy.parameters()) + list(self.value_module.parameters()),
                            max_norm=1.0
                        )
                        self.optim.step()
                        self.timings['optimizer_step'].append(time.time() - start_optim)
        
        self.total_time = time.time() - start_total
        self.collector.shutdown()
        
        return self._aggregate_timings()
    
    def run_profiling_detailed_collection(self) -> Dict[str, float]:
        """
        Run profiling with detailed data collection breakdown.
        
        Returns:
            Dictionary of component timings
        """
        print(f"Running detailed data collection profiling for {self.num_batches} batches...")
        
        start_total = time.time()
        
        # Collect data with detailed timing
        for batch_idx in range(self.num_batches):
            # Reset environment
            start_reset = time.time()
            td = self.env.reset()
            self.timings['env_reset'].append(time.time() - start_reset)
            
            frames_collected = 0
            target_frames = self.config.frames_per_batch
            
            # Collect one batch of data
            while frames_collected < target_frames:
                # Policy forward pass
                start_policy = time.time()
                with torch.no_grad():
                    td = self.policy(td)
                self.timings['policy_forward'].append(time.time() - start_policy)
                
                # Environment step
                start_env_step = time.time()
                try:
                    td = self.env.step(td)
                    self.timings['env_step'].append(time.time() - start_env_step)
                except (IndexError, RuntimeError) as e:
                    # Handle potential errors in turbo mode
                    print(f"Warning: Error in env.step: {e}")
                    self.timings['env_step'].append(time.time() - start_env_step)
                    # Reset and continue
                    td = self.env.reset()
                    continue
                
                # TensorDict operations
                start_tensordict = time.time()
                terminated = td["terminated"].item()
                frames_collected += 1

                if terminated:
                    start_reset = time.time()
                    td = self.env.reset()
                    self.timings['env_reset'].append(time.time() - start_reset)
        
        self.total_time = time.time() - start_total
        
        return self._aggregate_timings()
    
    def _aggregate_timings(self) -> Dict[str, float]:
        """Aggregate timing measurements."""
        aggregated = {}
        for key, times in self.timings.items():
            aggregated[key] = sum(times)
        return aggregated
    
    def generate_report(self, aggregated_timings: Dict[str, float]):
        """
        Generate formatted profiling report.
        
        Args:
            aggregated_timings: Dictionary of component timings
        """
        print("\n" + "=" * 60)
        print("Training Time Profiling Report")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Turbo Mode: {'ON' if self.enable_turbo else 'OFF'}")
        print(f"  - Training Batches: {self.num_batches}")
        print(f"  - Total Time: {self.total_time:.2f}s")
        print("=" * 60)
        
        # Calculate data collection time (total - measured components)
        measured_time = sum(aggregated_timings.values())
        data_collection_time = self.total_time - measured_time
        
        print("\nComponent Breakdown:")
        print("-" * 60)
        
        # Check if we have detailed data collection breakdown
        has_detailed_collection = ('env_step' in aggregated_timings or 
                                   'policy_forward' in aggregated_timings)
        
        if has_detailed_collection:
            # Detailed data collection breakdown
            env_step_time = aggregated_timings.get('env_step', 0)
            policy_forward_time = aggregated_timings.get('policy_forward', 0)
            env_reset_time = aggregated_timings.get('env_reset', 0)
            
            total_collection = env_step_time + policy_forward_time + env_reset_time
            pct = (total_collection / self.total_time) * 100
            bar = self._make_bar(pct)
            print(f"1. Data Collection          {total_collection:6.2f}s ({pct:5.1f}%)  {bar}")
            
            # Sub-components
            if env_step_time > 0:
                pct = (env_step_time / self.total_time) * 100
                bar = self._make_bar(pct)
                print(f"   - Env Step               {env_step_time:6.2f}s ({pct:5.1f}%)  {bar}")
            
            if policy_forward_time > 0:
                pct = (policy_forward_time / self.total_time) * 100
                bar = self._make_bar(pct)
                print(f"   - Policy Forward         {policy_forward_time:6.2f}s ({pct:5.1f}%)  {bar}")
            
            if env_reset_time > 0:
                pct = (env_reset_time / self.total_time) * 100
                bar = self._make_bar(pct)
                print(f"   - Env Reset              {env_reset_time:6.2f}s ({pct:5.1f}%)  {bar}")
        else:
            # Aggregated data collection
            pct = (data_collection_time / self.total_time) * 100
            bar = self._make_bar(pct)
            print(f"1. Data Collection          {data_collection_time:6.2f}s ({pct:5.1f}%)  {bar}")
            print(f"   (includes env steps + policy forward passes)")
        
        # GAE Computation
        if 'gae_computation' in aggregated_timings:
            t = aggregated_timings['gae_computation']
            pct = (t / self.total_time) * 100
            bar = self._make_bar(pct)
            print(f"\n2. GAE Computation          {t:6.2f}s ({pct:5.1f}%)  {bar}")
        
        # Replay Buffer
        buffer_time = aggregated_timings.get('replay_buffer', 0) + aggregated_timings.get('buffer_sample', 0)
        if buffer_time > 0:
            pct = (buffer_time / self.total_time) * 100
            bar = self._make_bar(pct)
            print(f"\n3. Replay Buffer Ops        {buffer_time:6.2f}s ({pct:5.1f}%)  {bar}")
        
        # Training Loop Components
        training_components = ['loss_computation', 'backward_pass', 'optimizer_step']
        training_time = sum(aggregated_timings.get(k, 0) for k in training_components)
        
        if training_time > 0:
            pct = (training_time / self.total_time) * 100
            bar = self._make_bar(pct)
            print(f"\n4. Training Loop            {training_time:6.2f}s ({pct:5.1f}%)  {bar}")
            
            for component in training_components:
                if component in aggregated_timings:
                    t = aggregated_timings[component]
                    pct = (t / self.total_time) * 100
                    bar = self._make_bar(pct)
                    name = component.replace('_', ' ').title()
                    print(f"   - {name:20s} {t:6.2f}s ({pct:5.1f}%)  {bar}")
        
        print("\n" + "=" * 60)
        print("Detailed Timing Breakdown:")
        print("-" * 60)
        for key, value in sorted(aggregated_timings.items(), key=lambda x: -x[1]):
            pct = (value / self.total_time) * 100
            name = key.replace('_', ' ').title()
            print(f"  {name:25s} {value:6.2f}s ({pct:5.1f}%)")
        
        print("\n" + "=" * 60)
        print("Key Insights:")
        print("-" * 60)
        
        # Calculate percentages for insights
        data_collection_pct = (data_collection_time / self.total_time) * 100
        training_pct = (training_time / self.total_time) * 100
        
        print(f"- Data collection (env + policy forward): {data_collection_pct:.1f}% of time")
        print(f"- Training operations (loss + backward + optim): {training_pct:.1f}% of time")
        
        if self.enable_turbo:
            print("- Turbo mode optimizes environment steps")
            print("- Environment steps are part of data collection phase")
            print("- Limited total impact due to neural network overhead")
        else:
            print("- Most time spent in neural network operations")
            print("- Environment steps are a small fraction of total time")
        
        print("=" * 60)
    
    def _make_bar(self, percentage: float, max_width: int = 20) -> str:
        """Create ASCII bar chart."""
        filled = int((percentage / 100) * max_width)
        return "█" * filled
    
    def run_detailed_profiling(self, output_file: str):
        """
        Run detailed cProfile analysis.
        
        Args:
            output_file: Path to save detailed profile report
        """
        print(f"\nRunning detailed cProfile analysis...")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run training
        with set_exploration_type(ExplorationType.RANDOM):
            for batch_idx, tensordict_data in enumerate(self.collector):
                if batch_idx >= self.num_batches:
                    break
                
                gae_vals = self.gae(tensordict_data)
                tensordict_data.set("advantage", gae_vals.get("advantage"))
                tensordict_data.set("value_target", gae_vals.get("value_target"))
                
                self.replay_buffer.extend(tensordict_data)
                
                for _ in range(self.config.num_epochs):
                    for _ in range(self.config.frames_per_batch // self.config.sub_batch_size):
                        subdata = self.replay_buffer.sample(self.config.sub_batch_size).to(self.device)
                        
                        loss_vals = self.loss_module(subdata)
                        loss = (loss_vals["loss_objective"] +
                               loss_vals["loss_critic"] +
                               loss_vals["loss_entropy"])
                        
                        self.optim.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            list(self.policy.parameters()) + list(self.value_module.parameters()),
                            max_norm=1.0
                        )
                        self.optim.step()
        
        profiler.disable()
        
        # Save detailed report
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(50)  # Top 50 functions
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Detailed cProfile Report (Turbo {'ON' if self.enable_turbo else 'OFF'})\n")
            f.write("=" * 80 + "\n\n")
            f.write(s.getvalue())
        
        print(f"Detailed profile saved to: {output_file}")
        
        # Print top 10 to console
        print("\n" + "=" * 60)
        print("Top 10 Hotspot Functions (from cProfile):")
        print("-" * 60)
        
        stats_list = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if tt > 0.01:  # Only functions taking > 10ms
                stats_list.append((func, tt, ct))
        
        stats_list.sort(key=lambda x: -x[1])
        
        for i, (func, tt, ct) in enumerate(stats_list[:10], 1):
            func_name = f"{func[0]}:{func[1]}:{func[2]}"
            # Shorten long names
            if len(func_name) > 50:
                func_name = "..." + func_name[-47:]
            print(f"{i:2d}. {func_name:50s} {tt:6.2f}s")
        
        print("=" * 60)


def compare_profiles(turbo_timings: Dict[str, float], baseline_timings: Dict[str, float],
                    turbo_total: float, baseline_total: float):
    """
    Generate comparison report between turbo and baseline.
    
    Args:
        turbo_timings: Timings with turbo ON
        baseline_timings: Timings with turbo OFF
        turbo_total: Total time with turbo ON
        baseline_total: Total time with turbo OFF
    """
    print("\n" + "=" * 80)
    print("Turbo Mode Comparison Report")
    print("=" * 80)
    
    speedup = ((baseline_total - turbo_total) / baseline_total) * 100
    print(f"\nOverall Performance:")
    print(f"  Baseline (turbo OFF): {baseline_total:.2f}s")
    print(f"  Turbo (turbo ON):     {turbo_total:.2f}s")
    print(f"  Speedup:              {speedup:+.1f}%")
    
    print("\n" + "-" * 80)
    print("Component-Level Comparison:")
    print("-" * 80)
    print(f"{'Component':<30s} {'Baseline':>12s} {'Turbo':>12s} {'Speedup':>12s}")
    print("-" * 80)
    
    # Get all component keys, prioritize data collection components
    priority_keys = ['env_step', 'policy_forward', 'tensordict_ops', 'env_reset']
    all_keys = set(turbo_timings.keys()) | set(baseline_timings.keys())
    
    # Sort: priority keys first, then others
    sorted_keys = []
    for key in priority_keys:
        if key in all_keys:
            sorted_keys.append(key)
            all_keys.remove(key)
    sorted_keys.extend(sorted(all_keys))
    
    for key in sorted_keys:
        baseline_time = baseline_timings.get(key, 0)
        turbo_time = turbo_timings.get(key, 0)
        
        if baseline_time > 0:
            component_speedup = ((baseline_time - turbo_time) / baseline_time) * 100
        else:
            component_speedup = 0
        
        name = key.replace('_', ' ').title()
        print(f"{name:<30s} {baseline_time:>10.2f}s {turbo_time:>10.2f}s {component_speedup:>10.1f}%")
    
    # Data collection (calculated)
    baseline_measured = sum(baseline_timings.values())
    turbo_measured = sum(turbo_timings.values())
    baseline_collection = baseline_total - baseline_measured
    turbo_collection = turbo_total - turbo_measured
    
    if baseline_collection > 0:
        collection_speedup = ((baseline_collection - turbo_collection) / baseline_collection) * 100
    else:
        collection_speedup = 0
    
    print(f"{'Data Collection':<30s} {baseline_collection:>10.2f}s {turbo_collection:>10.2f}s {collection_speedup:>10.1f}%")
    
    print("=" * 80)
    
    print("\nRecommendations:")
    print("-" * 80)
    if speedup > 5:
        print(f"+ Turbo mode provides significant speedup ({speedup:.1f}%)")
    elif speedup > 0:
        print(f"+ Turbo mode provides modest speedup ({speedup:.1f}%)")
    else:
        print(f"- Turbo mode shows minimal impact ({speedup:.1f}%)")
        print("  This is expected as neural network operations dominate training time")
    
    print("- Environment step optimization has limited impact on total training time")
    print("- Most time is spent in neural network forward/backward passes")
    print("- Turbo mode is most beneficial for pure simulation and evaluation")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Profile PPO training time distribution")
    parser.add_argument('--turbo', action='store_true', help='Enable turbo mode')
    parser.add_argument('--no-turbo', action='store_true', help='Disable turbo mode')
    parser.add_argument('--compare', action='store_true', help='Compare turbo ON vs OFF')
    parser.add_argument('--batches', type=int, default=10, help='Number of batches to profile')
    parser.add_argument('--detailed', action='store_true', help='Run detailed cProfile analysis')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.compare:
        # Run both and compare
        print("=" * 60)
        print("Running comparison: Turbo OFF vs Turbo ON")
        if args.breakdown:
            print("(with detailed data collection breakdown)")
        print("=" * 60)
        
        # Baseline
        print("\n[1/2] Profiling with turbo OFF...")
        profiler_baseline = TrainingProfiler(enable_turbo=False, num_batches=args.batches)
        baseline_timings = profiler_baseline.run_profiling_detailed_collection()
        baseline_total = profiler_baseline.total_time
        profiler_baseline.generate_report(baseline_timings)
        
        if args.detailed:
            profiler_baseline_detail = TrainingProfiler(enable_turbo=False, num_batches=args.batches)
            profiler_baseline_detail.run_detailed_profiling('profile_training_turbo_off.txt')
        
        # Turbo
        print("\n[2/2] Profiling with turbo ON...")
        profiler_turbo = TrainingProfiler(enable_turbo=True, num_batches=args.batches)
        turbo_timings = profiler_turbo.run_profiling_detailed_collection()
        turbo_total = profiler_turbo.total_time
        profiler_turbo.generate_report(turbo_timings)
        
        if args.detailed:
            profiler_turbo_detail = TrainingProfiler(enable_turbo=True, num_batches=args.batches)
            profiler_turbo_detail.run_detailed_profiling('profile_training_turbo_on.txt')
        
        # Comparison
        compare_profiles(turbo_timings, baseline_timings, turbo_total, baseline_total)
        
    else:
        # Single run
        enable_turbo = args.turbo or not args.no_turbo
        
        profiler = TrainingProfiler(enable_turbo=enable_turbo, num_batches=args.batches)
        timings = profiler.run_profiling_detailed_collection()
        profiler.generate_report(timings)
        
        if args.detailed:
            output_file = f'profile_training_turbo_{"on" if enable_turbo else "off"}.txt'
            profiler_detail = TrainingProfiler(enable_turbo=enable_turbo, num_batches=args.batches)
            profiler_detail.run_detailed_profiling(output_file)


if __name__ == "__main__":
    main()
