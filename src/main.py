"""Main entry point for mad-money data fetching."""

import asyncio

from src.data import Verbosity
from src.data.ohlcv import CoinbaseDataClient

FAV_SYMBOLS = [
                "AAVE/USDC",
                "ADA/USDC",
                "ALGO/USDC",
                "APE/USDC",
                "APT/USDC",
                "ARB/USDC",
                "ATOM/USDC",
                "AVAX/USDC",
                "BCH/USDC",
                "BONK/USDC",
                "BTC/USDC",
                "CHZ/USDC",
                "COMP/USDC",
                "CRV/USDC",
                "DOGE/USDC",
                "DOT/USDC",
                "EOS/USDC",
                "ETH/USDC",
                "FIL/USDC",
                "GRT/USDC",
                "ICP/USDC",
                "INJ/USDC",
                "LDO/USDC",
                "LINK/USDC",
                "LTC/USDC",
                "MANA/USDC",
                "MKR/USDC",
                "NEAR/USDC",
                "OP/USDC",
                "QNT/USDC",
                "RNDR/USDC",
                "SAND/USDC",
                "SHIB/USDC",
                "SOL/USDC",
                "SNX/USDC",
                "TAO/USDC",
                "UNI/USDC",
                "XLM/USDC",
                "XRP/USDC",
                "XTZ/USDC",
                "ZEC/USDC"
]

TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "6h",
    "1d"
]


async def main():
    """Fetch OHLCV data for multiple symbols and timeframes."""
    async with CoinbaseDataClient(
        verbosity=Verbosity.PROGRESS,
        max_concurrency=14,
        min_request_interval=0.03,
        batch_concurrency=4,
        batch_queue_size=12,
        enable_intra_combo_concurrency=True,
    ) as client:
        await client.fetch_multiple_and_save(
            symbols= FAV_SYMBOLS,
            timeframes= ["1d"],
            start_date="2026-01-01",
            end_date="2026-02-28",
        )


if __name__ == "__main__":
    asyncio.run(main())
