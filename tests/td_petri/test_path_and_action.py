"""
Unit tests for PathRegistry and ActionSpaceBuilder.

Tests the single source of truth for route definitions and action space construction.
"""

import pytest
from solutions.Td_petri.rl.path_registry import PathRegistry
from solutions.Td_petri.rl.action_space import ActionSpaceBuilder


class TestPathRegistry:
    """Test PathRegistry class."""
    
    def test_path_registry_creation(self):
        """Test creating a PathRegistry."""
        registry = PathRegistry()
        assert registry.pathC is not None
        assert registry.pathD is not None
    
    def test_path_c_structure(self):
        """Test Route C structure."""
        registry = PathRegistry()
        pathC = registry.pathC
        
        # Should have multiple stages
        assert len(pathC) > 0
        # First stage should be LP1 -> AL
        assert 'LP1' in pathC[0][0][0]
        assert 'AL' in pathC[0][0][0]
    
    def test_path_d_structure(self):
        """Test Route D structure."""
        registry = PathRegistry()
        pathD = registry.pathD
        
        # Should have multiple stages
        assert len(pathD) > 0
        # First stage should be LP2 -> AL
        assert 'LP2' in pathD[0][0][0]
        assert 'AL' in pathD[0][0][0]
    
    def test_get_all_paths(self):
        """Test getting all paths."""
        registry = PathRegistry()
        all_paths = registry.get_all_paths()
        
        assert 'C' in all_paths
        assert 'D' in all_paths
        assert all_paths['C'] == registry.pathC
        assert all_paths['D'] == registry.pathD
    
    def test_get_path_indices(self):
        """Test converting path names to indices."""
        registry = PathRegistry()
        
        # Create a mock id2t_name list
        id2t_name = [
            'ARM1_PICK__LP1__TO__AL',
            'ARM1_MOVE__LP1__TO__AL',
            'ARM1_LOAD__LP1__TO__AL',
            'PROC__AL',
            'ARM1_PICK__LP2__TO__AL',
            'ARM1_MOVE__LP2__TO__AL',
            'ARM1_LOAD__LP2__TO__AL',
        ]
        
        pathC_idx = registry.get_path_indices(id2t_name, 'C')
        
        # Should return indices instead of names
        assert isinstance(pathC_idx, list)
        assert len(pathC_idx) > 0
        # First stage, first branch should contain indices
        assert all(isinstance(idx, int) for idx in pathC_idx[0][0])


class TestActionSpaceBuilder:
    """Test ActionSpaceBuilder class."""
    
    def test_action_space_builder_creation(self):
        """Test creating an ActionSpaceBuilder."""
        registry = PathRegistry()
        builder = ActionSpaceBuilder(registry)
        
        assert builder.A > 0
        assert len(builder.aid2chain) == builder.A
        assert len(builder.chain2aid) > 0
    
    def test_action_space_deduplication(self):
        """Test that shared chains are deduplicated."""
        registry = PathRegistry()
        builder = ActionSpaceBuilder(registry)
        
        # Chains shared between routes should only appear once
        # For example, AL -> LLA_S2 is in both routes
        shared_chain_count = 0
        for tags in builder.aid2tags:
            if len(tags) > 1:  # Shared between multiple routes
                shared_chain_count += 1
        
        # There should be some shared chains
        assert shared_chain_count > 0
    
    def test_parallel_stage_tracking(self):
        """Test that parallel stages are correctly identified."""
        registry = PathRegistry()
        builder = ActionSpaceBuilder(registry)
        
        # Count parallel actions
        parallel_count = sum(builder.aid_is_parallel)
        
        # There should be parallel actions (PM7/PM8, PM1-4, PM9/PM10)
        assert parallel_count > 0
        
        # Check that parallel actions have valid pstage
        for aid in range(builder.A):
            if builder.aid_is_parallel[aid]:
                assert builder.aid_pstage[aid] >= 0
            else:
                assert builder.aid_pstage[aid] == -1
    
    def test_get_action_space_info(self):
        """Test getting complete action space information."""
        registry = PathRegistry()
        builder = ActionSpaceBuilder(registry)
        
        info = builder.get_action_space_info()
        
        assert 'aid2chain' in info
        assert 'chain2aid' in info
        assert 'aid_is_parallel' in info
        assert 'aid_pstage' in info
        assert 'aid2tags' in info
        assert 'A' in info
        
        assert info['A'] == len(info['aid2chain'])
    
    def test_get_parallel_stage_info(self):
        """Test getting parallel stage information."""
        registry = PathRegistry()
        builder = ActionSpaceBuilder(registry)
        
        stage_info = builder.get_parallel_stage_info()
        
        # Should have information about parallel stages
        assert len(stage_info) > 0
        
        # Each stage should have a count of chains
        for pstage, count in stage_info.items():
            assert isinstance(pstage, int)
            assert count > 0
    
    def test_chain_to_aid_mapping(self):
        """Test bidirectional mapping between chains and action IDs."""
        registry = PathRegistry()
        builder = ActionSpaceBuilder(registry)
        
        # Test that chain2aid and aid2chain are consistent
        for aid, chain in enumerate(builder.aid2chain):
            assert builder.chain2aid[chain] == aid
        
        for chain, aid in builder.chain2aid.items():
            assert builder.aid2chain[aid] == chain


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
