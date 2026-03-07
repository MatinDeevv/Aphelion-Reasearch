"""Tests for APHELION VWAP Features."""

from aphelion.features.vwap import VWAPCalculator


class TestVWAP:
    def test_initial_state(self):
        vwap = VWAPCalculator()
        d = vwap.to_dict()
        assert d["session_vwap"] == 0.0

    def test_single_bar(self):
        vwap = VWAPCalculator()
        vwap.update(high=2852, low=2848, close=2850, volume=100)
        d = vwap.to_dict()
        # Typical price = (2852+2848+2850)/3 = 2850
        assert abs(d["session_vwap"] - 2850.0) < 0.1

    def test_vwap_bands(self):
        vwap = VWAPCalculator()
        # Feed multiple bars to get standard deviation
        for i in range(20):
            vwap.update(2852 + i * 0.1, 2848 + i * 0.1, 2850 + i * 0.1, 100)
        d = vwap.to_dict()
        assert d["vwap_upper_1"] > d["session_vwap"]
        assert d["vwap_lower_1"] < d["session_vwap"]
        assert d["vwap_upper_2"] > d["vwap_upper_1"]

    def test_session_reset(self):
        vwap = VWAPCalculator()
        vwap.update(2852, 2848, 2850, 100)
        vwap.reset_session()
        vwap.update(2862, 2858, 2860, 100)
        d = vwap.to_dict()
        # After reset, VWAP should be near 2860
        assert abs(d["session_vwap"] - 2860.0) < 0.5

    def test_price_vs_vwap(self):
        vwap = VWAPCalculator()
        vwap.update(2852, 2848, 2855, 100)
        d = vwap.to_dict()
        # Close (2855) above VWAP → positive
        assert d["price_vs_vwap"] > 0
