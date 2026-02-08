"""
Performance tests for refactored TimedPetri.

Compares performance before and after refactoring to ensure no regression.
"""

import pytest
import time
import numpy as np
from solutions.Td_petri.tdpn import TimedPetri


class TestPerformance:
    """Performance tests for TimedPetri."""
    
    def test_initialization_time(self):
        """Test that initialization completes in reasonable time."""
        start = time.time()
        net = TimedPetri()
        elapsed = time.time() - start
        
        # Should initialize in less than 5 seconds
        assert elapsed < 5.0, f"Initialization took {elapsed:.2f}s, expected < 5s"
    
    def test_reset_time(self):
        """Test that reset completes in reasonable time."""
        net = TimedPetri()
        
        start = time.time()
        net.reset()
        elapsed = time.time() - start
        
        # Should reset in less than 1 second
        assert elapsed < 1.0, f"Reset took {elapsed:.2f}s, expected < 1s"
    
    def test_step_time(self):
        """Test that step completes in reasonable time."""
        net = TimedPetri()
        obs, mask = net.reset()
        
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]
        
        start = time.time()
        net.step(action)
        elapsed = time.time() - start
        
        # Should step in less than 0.5 seconds
        assert elapsed < 0.5, f"Step took {elapsed:.2f}s, expected < 0.5s"
    
    def test_episode_throughput(self):
        """Test throughput for a complete episode."""
        net = TimedPetri()
        
        start = time.time()
        obs, mask = net.reset()
        
        steps = 0
        done = False
        
        while not done and steps < 100:
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break
            
            action = valid_actions[0]
            mask, obs, time_val, done, reward = net.step(action)
            steps += 1
        
        elapsed = time.time() - start
        
        # Should complete 100 steps in less than 30 seconds
        assert elapsed < 30.0, f"100 steps took {elapsed:.2f}s, expected < 30s"
        
        # Calculate steps per second
        steps_per_sec = steps / elapsed
        print(f"\nThroughput: {steps_per_sec:.2f} steps/sec ({steps} steps in {elapsed:.2f}s)")
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable."""
        import sys
        
        net = TimedPetri()
        net.reset()
        
        # Check size of key data structures
        config_size = sys.getsizeof(net.config)
        path_registry_size = sys.getsizeof(net.path_registry)
        resource_mgr_size = sys.getsizeof(net.resource_mgr)
        
        # These should be reasonably sized (< 10KB each)
        assert config_size < 10000, f"Config size: {config_size} bytes"
        assert path_registry_size < 10000, f"PathRegistry size: {path_registry_size} bytes"
        assert resource_mgr_size < 10000, f"ResourceManager size: {resource_mgr_size} bytes"
    
    def test_observation_building_performance(self):
        """Test observation building performance."""
        net = TimedPetri()
        net.reset()
        
        # Build observation multiple times
        iterations = 1000
        start = time.time()
        
        for _ in range(iterations):
            obs = net.obs_builder.build_observation(net.marks)
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations
        
        # Should build observation in less than 1ms on average
        assert avg_time < 0.001, f"Observation building took {avg_time*1000:.2f}ms, expected < 1ms"
        print(f"\nObservation building: {avg_time*1000:.4f}ms per call")


class TestScalability:
    """Test scalability with different configurations."""
    
    def test_multiple_instances(self):
        """Test creating multiple instances."""
        start = time.time()
        
        instances = []
        for _ in range(10):
            net = TimedPetri()
            instances.append(net)
        
        elapsed = time.time() - start
        
        # Should create 10 instances in less than 30 seconds
        assert elapsed < 30.0, f"Creating 10 instances took {elapsed:.2f}s"
        print(f"\nCreated 10 instances in {elapsed:.2f}s ({elapsed/10:.2f}s per instance)")
    
    def test_parallel_episodes(self):
        """Test running multiple episodes in sequence."""
        net = TimedPetri()
        
        episode_times = []
        
        for episode in range(5):
            start = time.time()
            obs, mask = net.reset()
            
            steps = 0
            done = False
            
            while not done and steps < 50:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    break
                
                action = valid_actions[0]
                mask, obs, time_val, done, reward = net.step(action)
                steps += 1
            
            elapsed = time.time() - start
            episode_times.append(elapsed)
        
        avg_time = np.mean(episode_times)
        std_time = np.std(episode_times)
        
        print(f"\nEpisode times: {avg_time:.2f}s ± {std_time:.2f}s")
        
        # Performance should be consistent (std < 50% of mean)
        assert std_time < avg_time * 0.5, "Performance is inconsistent across episodes"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print statements
