# tests/test_config.py
from src.config import Cfg

def test_timeframe_seconds_basic():
    c = Cfg(timeframe="M5")
    assert c.timeframe_seconds() == 5 * 60
    c.timeframe = "H1"
    assert c.timeframe_seconds() == 3600
    c.timeframe = "D1"
    assert c.timeframe_seconds() == 86400
    c.timeframe = "X1"
    assert c.timeframe_seconds() is None
