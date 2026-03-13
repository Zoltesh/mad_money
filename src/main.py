"""Main entry point for mad-money data fetching."""

import asyncio

from src.data import Verbosity
from src.data.ohlcv import CoinbaseDataClient


async def main():
    """Fetch OHLCV data for multiple symbols and timeframes."""
    async with CoinbaseDataClient(verbosity=Verbosity.PROGRESS) as client:
        results = await client.fetch_multiple(
            symbols=["BTC/USDC", "ETH/USDC", "SOL/USDC"],  # Multiple assets
            timeframes=[
                "1m",
                "5m",
                "15m",
                "30m",
                "1h",
                "2h",
                "6h",
                "1d",
            ],  # Multiple timeframes
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        for symbol, timeframes in results.items():
            for timeframe, df in timeframes.items():
                client.save(df, symbol, timeframe)


if __name__ == "__main__":
    asyncio.run(main())
