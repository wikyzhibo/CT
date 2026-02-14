"""
Unit tests for interval utilities.

Tests interval operations and resource occupation management.
"""

import pytest
from solutions.Td_petri.resources.interval_utils import (
    Interval, ActionInfo, _first_free_time_at, _first_free_time_open, _insert_interval_sorted
)


class TestInterval:
    """Test Interval dataclass."""
    
    def test_interval_creation(self):
        """Test creating an Interval."""
        itv = Interval(start=10, end=20, tok_key=1)
        assert itv.start == 10
        assert itv.end == 20
        assert itv.tok_key == 1
    
    def test_interval_with_metadata(self):
        """Test Interval with additional metadata."""
        itv = Interval(
            start=10, end=20, tok_key=1,
            kind=0, from_loc="PM7", to_loc="LLC", wafer_type=1
        )
        assert itv.kind == 0
        assert itv.from_loc == "PM7"
        assert itv.to_loc == "LLC"
        assert itv.wafer_type == 1


class TestActionInfo:
    """Test ActionInfo dataclass."""
    
    def test_action_info_creation(self):
        """Test creating ActionInfo."""
        info = ActionInfo(
            t=5,
            fire_times=[10.0, 15.0, 20.0],
            t_name="ARM2_PICK__PM7__TO__LLC",
            chain=["ARM2_PICK__PM7__TO__LLC", "PROC__LLC"]
        )
        assert info.t == 5
        assert len(info.fire_times) == 3
        assert info.t_name == "ARM2_PICK__PM7__TO__LLC"
        assert len(info.chain) == 2


class TestInsertIntervalSorted:
    """Test _insert_interval_sorted function."""
    
    def test_insert_into_empty_list(self):
        """Test inserting into empty list."""
        intervals = []
        itv = Interval(start=10, end=20, tok_key=1)
        _insert_interval_sorted(intervals, itv)
        
        assert len(intervals) == 1
        assert intervals[0].start == 10
    
    def test_insert_maintains_order(self):
        """Test that insertion maintains sorted order."""
        intervals = [
            Interval(start=10, end=20, tok_key=1),
            Interval(start=30, end=40, tok_key=2),
        ]
        
        new_itv = Interval(start=25, end=28, tok_key=3)
        _insert_interval_sorted(intervals, new_itv)
        
        assert len(intervals) == 3
        assert intervals[0].start == 10
        assert intervals[1].start == 25
        assert intervals[2].start == 30
    
    def test_insert_at_beginning(self):
        """Test inserting at the beginning."""
        intervals = [
            Interval(start=20, end=30, tok_key=1),
            Interval(start=40, end=50, tok_key=2),
        ]
        
        new_itv = Interval(start=5, end=10, tok_key=3)
        _insert_interval_sorted(intervals, new_itv)
        
        assert intervals[0].start == 5
        assert len(intervals) == 3
    
    def test_insert_at_end(self):
        """Test inserting at the end."""
        intervals = [
            Interval(start=10, end=20, tok_key=1),
            Interval(start=30, end=40, tok_key=2),
        ]
        
        new_itv = Interval(start=50, end=60, tok_key=3)
        _insert_interval_sorted(intervals, new_itv)
        
        assert intervals[2].start == 50
        assert len(intervals) == 3


class TestFirstFreeTimeAt:
    """Test _first_free_time_at function."""
    
    def test_empty_intervals(self):
        """Test with no existing intervals."""
        result = _first_free_time_at([], 10, 20)
        assert result == 10
    
    def test_zero_duration(self):
        """Test with zero duration."""
        intervals = [Interval(start=10, end=20, tok_key=1)]
        result = _first_free_time_at(intervals, 15, 15)
        assert result == 15
    
    def test_no_conflict(self):
        """Test when there's no conflict."""
        intervals = [Interval(start=10, end=20, tok_key=1)]
        result = _first_free_time_at(intervals, 25, 35)
        assert result == 25
    
    def test_conflict_pushes_time(self):
        """Test when conflict pushes start time forward."""
        intervals = [Interval(start=10, end=20, tok_key=1)]
        result = _first_free_time_at(intervals, 15, 25)
        # Should be pushed to 20 (end of conflicting interval)
        assert result == 20
    
    def test_multiple_conflicts(self):
        """Test with multiple conflicting intervals."""
        intervals = [
            Interval(start=10, end=20, tok_key=1),
            Interval(start=25, end=35, tok_key=2),
            Interval(start=40, end=50, tok_key=3),
        ]
        result = _first_free_time_at(intervals, 15, 45)
        # Should be pushed past all conflicts to 50
        assert result == 50
    
    def test_previous_interval_covers_start(self):
        """Test when previous interval covers the start time."""
        intervals = [
            Interval(start=5, end=15, tok_key=1),
            Interval(start=20, end=30, tok_key=2),
        ]
        result = _first_free_time_at(intervals, 10, 18)
        # Should start at 15 (end of first interval)
        assert result == 15


class TestFirstFreeTimeOpen:
    """Test _first_free_time_open function."""
    
    def test_empty_intervals(self):
        """Test with no existing intervals."""
        result = _first_free_time_open([], 10)
        assert result == 10
    
    def test_no_overlap(self):
        """Test when no interval overlaps."""
        intervals = [Interval(start=5, end=10, tok_key=1)]
        result = _first_free_time_open(intervals, 15)
        assert result == 15
    
    def test_overlap_pushes_time(self):
        """Test when interval overlaps the start time."""
        intervals = [Interval(start=5, end=20, tok_key=1)]
        result = _first_free_time_open(intervals, 10)
        assert result == 20
    
    def test_multiple_intervals(self):
        """Test with multiple intervals."""
        intervals = [
            Interval(start=5, end=15, tok_key=1),
            Interval(start=20, end=30, tok_key=2),
        ]
        result = _first_free_time_open(intervals, 25)
        # Should be pushed to 30
        assert result == 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
