"""Progress tracking utilities for OHLCV data fetching."""

from __future__ import annotations

import os
import sys
from math import ceil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.ohlcv import Verbosity

# Import rich - will be installed as dependency
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.style import Style


def _is_test_environment() -> bool:
    """Detect if running in a test environment.

    Checks for:
    - TEST env var set to "1" or "true"
    - pytest in sys.modules
    - pytest in sys.argv

    Returns:
        True if running in test environment, False otherwise.
    """
    # Check TEST environment variable
    test_env = os.environ.get("TEST", "").lower()
    if test_env in ("1", "true"):
        return True

    # Check if pytest is in sys.modules
    if "pytest" in sys.modules:
        return True

    # Check if pytest is in command line arguments
    for arg in sys.argv:
        if "pytest" in arg:
            return True

    return False


# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "6h": 21600,
    "1d": 86400,
}

# Color palette for progress bars (Rich styles)
PROGRESS_COLORS = [
    "blue",
    "green",
    "yellow",
    "cyan",
    "magenta",
    "red",
    "bright_black",
    "white",
]


def build_shared_progress(progress_class: Any) -> Progress:
    """Create configured shared progress instance for batch fetches."""
    return progress_class(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} batches)"),
        TimeRemainingColumn(),
    )


def format_activity_description(
    active_requests: int,
    completed_batches: int,
    total_batches: int,
    failed_batches: int,
) -> str:
    """Build a readable shared activity status line."""
    return (
        "[bold white]ALL[/bold white] "
        f"active={active_requests} "
        f"done={completed_batches}/{total_batches} "
        f"failed={failed_batches}"
    )


def create_activity_state(
    shared_progress: Progress, total_batches: int
) -> dict[str, Any]:
    """Create activity tracking state and add the aggregate progress task."""
    normalized_total = max(1, total_batches)
    initial_description = format_activity_description(
        active_requests=0,
        completed_batches=0,
        total_batches=normalized_total,
        failed_batches=0,
    )
    task_id = shared_progress.add_task(initial_description, total=normalized_total)
    return {
        "task_id": task_id,
        "active": 0,
        "completed": 0,
        "failed": 0,
        "total": normalized_total,
    }


def calculate_expected_batches(start_ts: int, end_ts: int, timeframe: str) -> int:
    """Calculate the expected number of API batches needed.

    Args:
        start_ts: Start timestamp in milliseconds.
        end_ts: End timestamp in milliseconds.
        timeframe: Timeframe string (e.g., "1m", "5m", "1h").

    Returns:
        Expected number of batches needed.

    Raises:
        ValueError: If timeframe is not recognized.
    """
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unknown timeframe: {timeframe}. Supported: {list(TIMEFRAME_SECONDS.keys())}"
        )

    # Convert milliseconds to seconds
    total_seconds = (end_ts - start_ts) / 1000
    timeframe_seconds = TIMEFRAME_SECONDS[timeframe]

    # Formula: ceil(total_seconds / (timeframe_seconds * 300))
    candles_per_batch = timeframe_seconds * 300
    batches = ceil(total_seconds / candles_per_batch)

    return max(1, batches)


def get_progress_color(symbol: str, timeframe: str) -> str:
    """Get a consistent color name for a symbol/timeframe combination.

    Uses hash to consistently select from the color palette.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USD").
        timeframe: Timeframe string (e.g., "1m", "1h").

    Returns:
        Rich color name string.
    """
    # Create consistent hash from symbol and timeframe
    key = f"{symbol}:{timeframe}"
    hash_value = hash(key)
    index = abs(hash_value) % len(PROGRESS_COLORS)
    return PROGRESS_COLORS[index]


class RichProgressManager:
    """Rich-based progress manager for OHLCV data fetching.

    Provides progress tracking using Rich's Progress class, which properly
    handles concurrent updates in both terminal and JupyterLab environments.
    """

    def __init__(
        self,
        total: int,
        symbol: str,
        timeframe: str,
        verbosity: Verbosity | str = "disabled",
    ) -> None:
        """Initialize the Rich progress manager.

        Args:
            total: Total number of batches.
            symbol: Trading pair symbol for display.
            timeframe: Timeframe for display.
            verbosity: Verbosity level - if DISABLED, this is a no-op.
        """
        # Import Verbosity at runtime to avoid circular import
        from src.data.ohlcv import Verbosity

        self.total = total
        self.symbol = symbol
        self.timeframe = timeframe
        self.verbosity = verbosity
        self._progress: Progress | None = None
        self._task_id: Any = None

        # Handle both enum and string inputs
        if isinstance(verbosity, Verbosity):
            self._enabled = verbosity != Verbosity.DISABLED
        else:
            self._enabled = verbosity != "disabled"

    def start(self) -> None:
        """Start the progress bar."""
        if not self._enabled:
            return

        # Rich auto-detects terminal vs Jupyter, no manual detection needed
        color = get_progress_color(self.symbol, self.timeframe)

        # Create progress with columns for batch progress
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn(
                f"[{color}]{{task.description}}",
                style=Style(color=color),
            ),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} batches)"),
            TimeRemainingColumn(),
            console=None,  # Auto-detect terminal vs Jupyter
        )

        # Start the progress display
        self._progress.start()

        # Add this task to the progress
        description = f"{self.symbol} {self.timeframe}"
        self._task_id = self._progress.add_task(description, total=self.total)

    def update(
        self, n: int = 1, candles_so_far: int = 0, total_candles: int = 0
    ) -> None:
        """Update the progress bar.

        Args:
            n: Number of batches to increment.
            candles_so_far: Number of candles fetched so far.
            total_candles: Estimated total candles expected.
        """
        if self._progress is None or self._task_id is None:
            return

        # Update the task progress
        self._progress.update(
            self._task_id,
            advance=n,
        )

    def close(self) -> None:
        """Close the progress bar."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None


# Alias for backward compatibility
ProgressTracker = RichProgressManager
