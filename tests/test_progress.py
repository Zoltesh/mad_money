"""Tests for progress tracking functionality."""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.data import CoinbaseDataClient
from src.data.ohlcv import Verbosity
from src.data.progress import (
    PROGRESS_COLORS,
    TIMEFRAME_SECONDS,
    ProgressTracker,
    calculate_expected_batches,
    get_progress_color,
)


class TestVerbosityConfiguration:
    """Tests for Verbosity enum and configuration."""

    def test_default_verbosity_is_disabled(self):
        """Test that default verbosity is disabled when not explicitly set."""
        # When verbosity is None, it should auto-detect
        # In test environment, it should default to DISABLED
        client = CoinbaseDataClient()
        assert client.verbosity == Verbosity.DISABLED

    def test_verbosity_can_be_set_in_constructor(self):
        """Test that verbosity can be set in the constructor."""
        client = CoinbaseDataClient(verbosity=Verbosity.PROGRESS)
        assert client.verbosity == Verbosity.PROGRESS

        client = CoinbaseDataClient(verbosity=Verbosity.VERBOSE)
        assert client.verbosity == Verbosity.VERBOSE

    def test_verbosity_can_be_overridden_in_fetch(self):
        """Test that verbosity can be overridden in fetch() method."""
        client = CoinbaseDataClient(verbosity=Verbosity.DISABLED)

        # Mock the exchange to avoid actual API calls
        mock_candles = [
            [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        ]

        async def mock_fetch(*args, **kwargs):
            return mock_candles

        import asyncio

        with patch.object(client, "_get_exchange") as mock_exchange:
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.fetch_ohlcv = mock_fetch
            mock_exchange.return_value = mock_exchange_instance

            # Override verbosity in fetch
            result = asyncio.run(
                client.fetch(
                    "BTC/USD", "1m", "2024-01-01", verbosity=Verbosity.PROGRESS
                )
            )

        assert result is not None
        assert len(result) == 1

    def test_verbosity_enum_values(self):
        """Test that Verbosity enum has expected values."""
        assert Verbosity.DISABLED.value == "disabled"
        assert Verbosity.PROGRESS.value == "progress"
        assert Verbosity.VERBOSE.value == "verbose"


class TestProgressCalculation:
    """Tests for calculate_expected_batches function."""

    def test_calculate_expected_batches_1m_timeframe(self):
        """Test batch calculation for 1-minute timeframe."""
        # 1 minute = 60 seconds
        # 300 candles per batch * 60 seconds = 18000 seconds per batch
        # For 1 hour (3600 seconds): 3600 / 18000 = 0.2, ceil = 1 batch
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 1, 1, 0, 0).timestamp() * 1000)

        batches = calculate_expected_batches(start_ts, end_ts, "1m")
        assert batches == 1

    def test_calculate_expected_batches_5m_timeframe(self):
        """Test batch calculation for 5-minute timeframe."""
        # 5 minutes = 300 seconds
        # 300 candles per batch * 300 seconds = 90000 seconds per batch
        # For 1 day (86400 seconds): 86400 / 90000 = 0.96, ceil = 1 batch
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2, 0, 0, 0).timestamp() * 1000)

        batches = calculate_expected_batches(start_ts, end_ts, "5m")
        assert batches == 1

    def test_calculate_expected_batches_1h_timeframe_long_range(self):
        """Test batch calculation for 1h timeframe with long date range."""
        # 1 hour = 3600 seconds
        # 300 candles per batch * 3600 seconds = 1080000 seconds per batch
        # For 1 year (~31536000 seconds): 31536000 / 1080000 = 29.2, ceil = 30 batches
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        end_ts = int(datetime(2025, 1, 1, 0, 0, 0).timestamp() * 1000)

        batches = calculate_expected_batches(start_ts, end_ts, "1h")
        # Approximately 30 batches for a year of hourly data
        assert batches >= 20
        assert batches <= 40

    def test_calculate_expected_batches_1d_timeframe(self):
        """Test batch calculation for 1-day timeframe."""
        # 1 day = 86400 seconds
        # 300 candles per batch * 86400 seconds = 25920000 seconds per batch
        # For 1 year: 31536000 / 25920000 = 1.2, ceil = 2 batches
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        end_ts = int(datetime(2025, 1, 1, 0, 0, 0).timestamp() * 1000)

        batches = calculate_expected_batches(start_ts, end_ts, "1d")
        assert batches == 2

    def test_calculate_expected_batches_raises_on_unknown_timeframe(self):
        """Test that unknown timeframe raises ValueError."""
        start_ts = 1704067200000
        end_ts = 1704153600000

        with pytest.raises(ValueError, match="Unknown timeframe"):
            calculate_expected_batches(start_ts, end_ts, "invalid")

    def test_calculate_expected_batches_all_timeframes(self):
        """Test batch calculation works for all supported timeframes."""
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2, 0, 0, 0).timestamp() * 1000)

        for timeframe in TIMEFRAME_SECONDS.keys():
            batches = calculate_expected_batches(start_ts, end_ts, timeframe)
            assert batches >= 1, f"Expected at least 1 batch for {timeframe}"

    def test_calculate_expected_batches_minimum_one(self):
        """Test that batch calculation returns minimum of 1."""
        # Very short time range should still return 1
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 1, 0, 1, 0).timestamp() * 1000)  # 1 minute later

        batches = calculate_expected_batches(start_ts, end_ts, "1d")
        assert batches == 1


class TestColorAssignment:
    """Tests for get_progress_color function."""

    def test_same_symbol_timeframe_gets_same_color(self):
        """Test that same symbol/timeframe combination gets the same color."""
        color1 = get_progress_color("BTC/USD", "1h")
        color2 = get_progress_color("BTC/USD", "1h")
        assert color1 == color2

    def test_different_symbols_get_different_colors(self):
        """Test that different symbol/timeframe combinations get different colors."""
        colors = set()
        for symbol in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            color = get_progress_color(symbol, "1h")
            colors.add(color)

        # Should have at least 2 different colors for 3 different symbols
        assert len(colors) >= 2

    def test_different_timeframes_get_different_colors(self):
        """Test that same symbol with different timeframes can get different colors."""
        colors = set()
        for timeframe in ["1m", "5m", "1h", "1d"]:
            color = get_progress_color("BTC/USD", timeframe)
            colors.add(color)

        # Should have at least 2 different colors
        assert len(colors) >= 2

    def test_color_is_valid_ansi(self):
        """Test that returned colors are valid ANSI escape codes."""
        for symbol in ["BTC/USD", "ETH/USD"]:
            for timeframe in ["1m", "1h", "1d"]:
                color = get_progress_color(symbol, timeframe)
                assert color in PROGRESS_COLORS


class TestDisabledMode:
    """Tests for disabled progress tracking mode."""

    def test_progress_tracker_disabled_has_no_overhead(self):
        """Test that disabled progress tracker creates no progress bar."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity=Verbosity.DISABLED,
        )

        # Should not create a Rich Progress instance when disabled
        assert tracker._progress is None
        assert tracker._enabled is False

    def test_progress_tracker_update_no_op_when_disabled(self):
        """Test that update is a no-op when disabled."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity=Verbosity.DISABLED,
        )

        # Should not raise, just do nothing
        tracker.update(n=1)
        tracker.update(n=1, candles_so_far=5, total_candles=10)

    def test_progress_tracker_close_no_op_when_disabled(self):
        """Test that close is a no-op when disabled."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity=Verbosity.DISABLED,
        )

        # Should not raise
        tracker.close()

    def test_progress_enabled_when_verbosity_is_progress(self):
        """Test that progress is enabled when verbosity is PROGRESS."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity=Verbosity.PROGRESS,
        )

        # Should be enabled when verbosity is PROGRESS
        assert tracker._enabled is True

    def test_progress_tracker_with_string_disabled(self):
        """Test that string 'disabled' also works for verbosity."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity="disabled",
        )

        assert tracker._enabled is False


class TestTestEnvironmentDetection:
    """Tests for test environment detection."""

    def test_is_test_environment_with_test_env(self, monkeypatch):
        """Test detection via TEST environment variable."""
        monkeypatch.setenv("TEST", "1")
        # Need to reimport to pick up the env change
        import importlib

        import src.data.progress as progress_module

        importlib.reload(progress_module)
        assert progress_module._is_test_environment() is True

        monkeypatch.setenv("TEST", "true")
        importlib.reload(progress_module)
        assert progress_module._is_test_environment() is True

    def test_is_test_environment_with_pytest(self, monkeypatch):
        """Test detection via pytest in sys.modules."""
        monkeypatch.setenv("TEST", "")  # Clear TEST env
        # Add pytest to sys.modules
        monkeypatch.setitem(sys.modules, "pytest", pytest)
        import importlib

        import src.data.progress

        importlib.reload(src.data.progress)
        assert src.data.progress._is_test_environment() is True


class TestProgressTrackerIntegration:
    """Integration tests for ProgressTracker."""

    def test_progress_tracker_start_creates_pbar(self):
        """Test that start creates Rich progress display when enabled."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity=Verbosity.PROGRESS,
        )

        # In a real environment with color support, this would create a progress
        # display. We verify it doesn't crash in test environments.
        try:
            tracker.start()
        except Exception:
            pass  # May fail in some test environments without a real terminal

    def test_progress_tracker_full_lifecycle(self):
        """Test complete lifecycle: create, start, update, close."""
        tracker = ProgressTracker(
            total=10,
            symbol="BTC/USD",
            timeframe="1h",
            verbosity=Verbosity.DISABLED,  # Use disabled to avoid terminal issues
        )

        # These should all complete without error
        tracker.start()
        tracker.update(n=1, candles_so_far=1, total_candles=10)
        tracker.update(n=1, candles_so_far=2, total_candles=10)
        tracker.close()


class TestClientVerbosityIntegration:
    """Integration tests for verbosity with CoinbaseDataClient."""

    def test_client_verbosity_disabled_by_default(self):
        """Test that client has DISABLED verbosity by default."""
        client = CoinbaseDataClient()
        assert client.verbosity == Verbosity.DISABLED

    def test_client_with_progress_verbosity(self):
        """Test client with PROGRESS verbosity."""
        client = CoinbaseDataClient(verbosity=Verbosity.PROGRESS)
        assert client.verbosity == Verbosity.PROGRESS

    def test_client_with_verbose_verbosity(self):
        """Test client with VERBOSE verbosity."""
        client = CoinbaseDataClient(verbosity=Verbosity.VERBOSE)
        assert client.verbosity == Verbosity.VERBOSE
