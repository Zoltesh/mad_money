"""Timeframe utilities for technical indicators."""

from __future__ import annotations

from src.data.progress import TIMEFRAME_SECONDS


def parse_timeframe(timeframe: str) -> int:
    """Parse a timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., "5m", "1h", "15m").

    Returns:
        Timeframe in seconds.

    Raises:
        ValueError: If timeframe is not supported.
    """
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unknown timeframe: {timeframe}. "
            f"Supported: {list(TIMEFRAME_SECONDS.keys())}"
        )
    return TIMEFRAME_SECONDS[timeframe]


def timeframe_ratio(target: str, base: str) -> int:
    """Calculate how many base candles fit in a target timeframe.

    Args:
        target: Target timeframe (e.g., "15m").
        base: Base timeframe (e.g., "5m").

    Returns:
        Number of base candles needed to form one target candle.

    Raises:
        ValueError: If target is not a multiple of base.
    """
    target_sec = parse_timeframe(target)
    base_sec = parse_timeframe(base)

    if target_sec % base_sec != 0:
        raise ValueError(
            f"Target timeframe {target} ({target_sec}s) is not a multiple of "
            f"base timeframe {base} ({base_sec}s)"
        )

    return target_sec // base_sec


def is_base_timeframe(timeframe: str, candidate_base: str) -> bool:
    """Check if candidate_base is the base timeframe for the given timeframe.

    Args:
        timeframe: The timeframe to check (e.g., "15m").
        candidate_base: The candidate base timeframe (e.g., "5m").

    Returns:
        True if candidate_base is the base for the given timeframe.
    """
    try:
        ratio = timeframe_ratio(timeframe, candidate_base)
        return ratio >= 1
    except ValueError:
        return False
