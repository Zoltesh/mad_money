"""Main entry point for mad-money data fetching."""

import asyncio

from src.data import Verbosity
from src.data.ohlcv import CoinbaseDataClient


async def main():
    """Fetch OHLCV data for multiple symbols and timeframes."""
    async with CoinbaseDataClient(verbosity=Verbosity.PROGRESS) as client:
        results = await client.fetch_multiple(
            symbols=[
                "AVAX/USDC",
                "ADA/USDC",
                "BCH/USDC",
                "BTC/USDC",
                "ETH/USDC",
                "DOGE/USDC",
                "LINK/USDC",
                "LTC/USDC",
                "SHIB/USDC",
                "SOL/USDC",
                "XRP/USDC"
                ],
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
            end_date="2025-01-07",
        )

        for symbol, timeframes in results.items():
            for timeframe, df in timeframes.items():
                client.save(df, symbol, timeframe)


if __name__ == "__main__":
    asyncio.run(main())
