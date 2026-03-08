import pytest

@pytest.fixture
def make_core():
    # Stub for core creation
    pass

@pytest.mark.xfail(reason="source not yet merged")
class TestSentinelCore:
    def test_initial_state(self, make_core):
        pass

    def test_update_equity_sets_peak(self, make_core):
        pass

    def test_l3_triggers_on_10pct_drawdown(self, make_core):
        pass

    def test_l3_not_triggered_on_small_drop(self, make_core):
        pass

    @pytest.mark.asyncio
    async def test_l3_publishes_critical_event(self, make_core):
        pass

    def test_register_and_count_positions(self, make_core):
        pass

    def test_close_position_removes_it(self, make_core):
        pass

    def test_total_exposure_sums_positions(self, make_core):
        pass

    def test_trading_halted_during_l3(self, make_core):
        pass

    def test_status_dict_has_all_keys(self, make_core):
        pass
