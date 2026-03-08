import pytest

@pytest.fixture
def make_cb():
    pass

@pytest.mark.xfail(reason="source not yet merged")
class TestCircuitBreaker:
    def test_initial_state_normal(self, make_cb):
        pass

    @pytest.mark.asyncio
    async def test_l1_triggers_at_5pct(self, make_cb):
        pass
        
    @pytest.mark.asyncio
    async def test_l2_triggers_at_7_5pct(self, make_cb):
        pass
        
    @pytest.mark.asyncio
    async def test_l3_triggers_at_10pct(self, make_cb):
        pass
        
    def test_no_double_trigger_l3(self, make_cb):
        pass
        
    def test_reset_from_l1(self, make_cb):
        pass
        
    def test_cannot_reset_from_l3(self, make_cb):
        pass
        
    def test_apply_multiplier_scales_size(self, make_cb):
        pass
        
    def test_apply_multiplier_l3_returns_zero(self, make_cb):
        pass
        
    def test_summary_dict_keys(self, make_cb):
        pass
        
    def test_peak_never_decreases(self, make_cb):
        pass
