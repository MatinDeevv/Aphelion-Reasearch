import pytest

@pytest.fixture
def make_sentinel_stack():
    pass

@pytest.mark.xfail(reason="source not yet merged")
class TestSentinelIntegration:
    def test_valid_trade_gets_approved(self, make_sentinel_stack):
        pass

    def test_trade_rejected_missing_stop_loss(self, make_sentinel_stack):
        pass

    def test_trade_rejected_bad_rr(self, make_sentinel_stack):
        pass

    def test_trade_rejected_at_max_positions(self, make_sentinel_stack):
        pass

    @pytest.mark.asyncio
    async def test_l3_blocks_all_trades(self, make_sentinel_stack):
        pass

    def test_circuit_breaker_reduces_size_in_l2(self, make_sentinel_stack):
        pass

    @pytest.mark.asyncio
    async def test_sl_breach_publishes_critical_event(self, make_sentinel_stack):
        pass

    @pytest.mark.asyncio
    async def test_full_pipeline_valid_trade_then_sl_hit(self, make_sentinel_stack):
        pass

    def test_rejection_rate_tracked(self, make_sentinel_stack):
        pass

    def test_all_sentinel_rules_run_on_invalid_proposal(self, make_sentinel_stack):
        pass

    def test_trade_rejected_bad_confidence(self, make_sentinel_stack):
        pass

    def test_rejection_metrics_reset(self, make_sentinel_stack):
        pass

    def test_l1_reduces_size_by_half(self, make_sentinel_stack):
        pass

    def test_valid_trade_approved_with_warning(self, make_sentinel_stack):
        pass
