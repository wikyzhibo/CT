"""
Unit tests for PetriConfig module.

Tests configuration loading, saving, and default values.
"""

import pytest
import json
import tempfile
from pathlib import Path
from solutions.Td_petri.core.config import PetriConfig, ModuleSpec


class TestModuleSpec:
    """Test ModuleSpec dataclass."""
    
    def test_module_spec_creation(self):
        """Test creating a ModuleSpec."""
        spec = ModuleSpec(tokens=50, capacity=100)
        assert spec.tokens == 50
        assert spec.capacity == 100


class TestPetriConfig:
    """Test PetriConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PetriConfig.default()
        
        assert config.parallel_groups is not None
        assert "PM1_4" in config.parallel_groups
        assert config.modules is not None
        assert "LP1" in config.modules
        assert config.history_length == 50
        assert len(config.reward_weights) == 7
    
    def test_config_modules(self):
        """Test module specifications in config."""
        config = PetriConfig.default()
        
        # Check LP1 module
        assert config.modules["LP1"].tokens == 50
        assert config.modules["LP1"].capacity == 100
        
        # Check PM7 module
        assert config.modules["PM7"].tokens == 0
        assert config.modules["PM7"].capacity == 1
    
    def test_config_routes(self):
        """Test route definitions."""
        config = PetriConfig.default()
        
        assert len(config.routes) == 2
        # Route C (LP1)
        assert config.routes[0][0] == "LP1"
        # Route D (LP2)
        assert config.routes[1][0] == "LP2"
    
    def test_config_stage_capacity(self):
        """Test stage capacity configuration."""
        config = PetriConfig.default()
        
        assert config.stage_capacity[1] == 2  # PM7/PM8
        assert config.stage_capacity[3] == 4  # PM1-PM4
        assert config.stage_capacity[5] == 2  # PM9/PM10
    
    def test_config_processing_time(self):
        """Test processing time configuration."""
        config = PetriConfig.default()
        
        assert config.processing_time[1] == 70   # PM7/PM8
        assert config.processing_time[3] == 600  # PM1-PM4
        assert config.processing_time[5] == 200  # PM9/PM10
    
    def test_config_to_json(self):
        """Test saving configuration to JSON."""
        config = PetriConfig.default()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            config.to_json(temp_path)
            
            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'parallel_groups' in data
            assert 'modules' in data
            assert 'routes' in data
            assert data['history_length'] == 50
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_from_json(self):
        """Test loading configuration from JSON."""
        # Create a test config
        original_config = PetriConfig.default()
        original_config.history_length = 100
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            original_config.to_json(temp_path)
            
            # Load it back
            loaded_config = PetriConfig.from_json(temp_path)
            
            assert loaded_config.history_length == 100
            assert loaded_config.modules["LP1"].tokens == 50
            assert len(loaded_config.routes) == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_from_json_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            PetriConfig.from_json('nonexistent_file.json')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
