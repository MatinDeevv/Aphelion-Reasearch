import pytest

@pytest.fixture
def sizer():
    pass

@pytest.mark.xfail(reason="source not yet merged")
class TestPositionSizer:
    def test_kelly_hard_cap(self, sizer):
        pass

    def test_kelly_quarter_multiplier(self, sizer):
        pass
        
    def test_kelly_zero_on_negative(self, sizer):
        pass
        
    def test_kelly_handles_zero_avg_win(self, sizer):
        pass
        
    def test_compute_size_pct_capped(self, sizer):
        pass
        
    def test_compute_size_pct_with_confidence(self, sizer):
        pass
        
    def test_pct_to_lots_basic(self, sizer):
        pass
        
    def test_pct_to_lots_minimum(self, sizer):
        pass
        
    def test_validate_size_ok(self, sizer):
        pass
        
    def test_validate_size_exceeds_max(self, sizer):
        pass
        
    def test_validate_total_exposure_exceeded(self, sizer):
        pass
