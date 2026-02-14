"""
重构后 Timed Petri Net environment 的使用示例

TimedPetri 类现在直接继承 TorchRL 的 EnvBase，
不再需要单独的 wrapper 层。
"""

import numpy as np
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.core.config import PetriConfig


def example_torchrl_interface():
    """Example using TorchRL interface directly on TimedPetri."""
    print("=" * 60)
    print("Example 1: TorchRL Interface (Direct)")
    print("=" * 60)

    # TimedPetri 现在直接继承 EnvBase
    env = TimedPetri(device='cpu', reward_mode='progress', seed=42)
    
    print(f"Action space size: {env.A}")
    print(f"Observation space size: {env.obs_dim}")
    
    # Reset environment using TorchRL interface
    td = env._reset()
    print(f"\nInitial observation shape: {td['observation'].shape}")
    print(f"Initial action mask shape: {td['action_mask'].shape}")
    print(f"Number of valid actions: {td['action_mask'].sum().item()}")
    
    # Run a few steps
    for step in range(5):
        # Get valid actions
        valid_actions = torch.where(td['action_mask'])[0]
        if len(valid_actions) == 0:
            print("No valid actions available!")
            break
        
        # Select first valid action
        action = valid_actions[0]
        
        # Take step using TorchRL interface
        td_action = td.clone()
        td_action.set("action", action.unsqueeze(0))
        td_next = env._step(td_action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action.item()}")
        print(f"  Reward: {td_next['reward'].item():.2f}")
        print(f"  Time: {td_next['time'].item()}")
        print(f"  Done: {td_next['done'].item()}")
        
        if td_next['done'].item():
            print("Episode finished!")
            break
        
        td = td_next


def example_simple_interface():
    """Example using simple Gym-like interface (reset/step methods)."""
    print("\n" + "=" * 60)
    print("Example 2: Simple Interface (reset/step)")
    print("=" * 60)
    
    # Create environment
    env = TimedPetri()
    
    print(f"Action space size: {env.A}")
    print(f"Observation space size: {env.obs_dim}")
    
    # Reset environment using simple interface
    obs, mask = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial action mask shape: {mask.shape}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        # Get valid actions
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            print("No valid actions available!")
            break
        
        # Select first valid action
        action = valid_actions[0]
        
        # Take step using simple interface
        mask, obs, time, done, reward = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Time: {time}")
        print(f"  Done: {done}")
        
        if done:
            print("Episode finished!")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")


def example_custom_config():
    """Example using custom configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Create custom config
    config = PetriConfig.default()
    config.history_length = 100
    config.reward_weights = [0, 10, 30, 100, 800, 980, 1000]
    
    print(f"History length: {config.history_length}")
    print(f"Reward weights: {config.reward_weights}")
    
    # Create environment with custom config
    env = TimedPetri(config=config)
    
    print(f"Observation space size: {env.obs_dim}")
    
    # Reset and check
    obs, mask = env.reset()
    print(f"Observation shape: {obs.shape}")


if __name__ == '__main__':
    # Import torch only if using TorchRL interface
    try:
        import torch
        example_torchrl_interface()
    except ImportError:
        print("TorchRL not available, skipping TorchRL example")
    
    example_simple_interface()
    example_custom_config()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
