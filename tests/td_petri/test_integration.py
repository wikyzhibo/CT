"""
Integration tests for refactored TimedPetri class.

Tests that the refactored code works correctly with the new modules.
"""

import pytest
import numpy as np
from solutions.Td_petri.tdpn import TimedPetri
from solutions.Td_petri.core.config import PetriConfig


class TestTimedPetriIntegration:
    """TimedPetri 的集成测试"""
    
    def test_initialization_default_config(self):
        """Test TimedPetri 默认初始化测试"""
        net = TimedPetri()
        
        assert net.config is not None
        assert net.path_registry is not None
        assert net.resource_mgr is not None
        assert net.obs_builder is not None
        assert net.reward_calc is not None
        assert net.A > 0
    
    def test_initialization_custom_config(self):
        """Test TimedPetri 自定义初始化测试"""
        config = PetriConfig.default()
        config.history_length = 100
        
        net = TimedPetri(config)
        
        assert net.his_len == 100
        assert net.obs_builder.history_length == 100
    
    def test_reset_functionality(self):
        """Test TimedPetri 重置功能测试"""
        net = TimedPetri()
        obs, mask = net.reset()
        
        # Check observation shape
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == net.obs_dim
        
        # Check mask
        assert isinstance(mask, np.ndarray)
        assert mask.shape[0] == net.A
        assert mask.dtype == bool
        
        # Should have at least one valid action
        assert np.any(mask)
    
    def test_step_functionality(self):
        """Test TimedPetri 步进功能测试"""
        net = TimedPetri()
        obs, mask = net.reset()
        
        # Find a valid action
        valid_actions = np.where(mask)[0]
        assert len(valid_actions) > 0
        
        action = valid_actions[0]
        
        # Take a step
        new_mask, new_obs, time, done, reward = net.step(action)
        
        # Check outputs
        assert isinstance(new_obs, np.ndarray)
        assert new_obs.shape[0] == net.obs_dim
        assert isinstance(new_mask, np.ndarray)
        assert isinstance(time, (int, float))
        assert isinstance(done, bool)
        assert isinstance(reward, (int, float))
    
    def test_observation_history_update(self):
        """观察历史测试"""
        net = TimedPetri()
        obs, mask = net.reset()
        
        # Initial history should be zeros
        initial_history = net.obs_builder.his_a
        assert all(h == 0 for h in initial_history)
        
        # Take a step
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]
        net.step(action)
        
        # History should be updated
        updated_history = net.obs_builder.his_a
        assert updated_history[-1] == action
        assert updated_history[0] == 0  # Oldest should still be 0
    
    def test_resource_manager_integration(self):
        """资源管理器功能测试"""
        net = TimedPetri()
        net.reset()
        
        # Resource manager should be initialized
        assert len(net.resource_mgr.res_occ) > 0
        
        # Backward compatibility: res_occ should point to resource_mgr
        assert net.res_occ is net.resource_mgr.res_occ
    
    def test_path_registry_integration(self):
        """路径登记器功能测试"""
        net = TimedPetri()
        
        # PathRegistry should be created
        assert net.path_registry is not None
        
        # Paths should be available
        all_paths = net.path_registry.get_all_paths()
        assert 'C' in all_paths
        assert 'D' in all_paths
    
    def test_action_space_consistency(self):
        """测试动作空间一致"""
        net = TimedPetri()
        
        # aid2chain and chain2aid should be consistent
        for aid, chain in enumerate(net.aid2chain):
            assert net.chain2aid[chain] == aid
    
    def test_multiple_episodes(self):
        """测试多episodes."""
        net = TimedPetri()
        
        for episode in range(3):
            obs, mask = net.reset()
            
            steps = 0
            done = False
            
            while not done and steps < 100:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    break
                
                action = valid_actions[0]
                mask, obs, time, done, reward = net.step(action)
                steps += 1
            
            # Should complete some steps
            assert steps > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_no_config_parameter(self):
        """Test that TimedPetri() still works without config parameter."""
        net = TimedPetri()
        assert net is not None
    
    def test_reset_signature(self):
        """Test that reset() returns (obs, mask)."""
        net = TimedPetri()
        result = net.reset()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_step_signature(self):
        """Test that step() returns expected values."""
        net = TimedPetri()
        obs, mask = net.reset()
        
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]
        
        result = net.step(action)
        
        assert isinstance(result, tuple)
        assert len(result) == 5  # mask, obs, time, done, reward
    
    def test_attribute_access(self):
        """Test that old attributes are still accessible."""
        net = TimedPetri()
        
        # These should still exist for backward compatibility
        assert hasattr(net, 'res_occ')
        assert hasattr(net, 'open_mod_occ')
        assert hasattr(net, 'stage_c')
        assert hasattr(net, 'proc')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
