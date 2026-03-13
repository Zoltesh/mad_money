import asyncio
from src.data import Verbosity
from src.data.ohlcv import CoinbaseDataClient

client = CoinbaseDataClient()


async def main():
    results = await client.fetch_multiple(
        symbols=["BTC/USDC", "ETH/USDC", "SOL/USDC"],  # Multiple assets
        timeframes=['1m', '5m', '15m', '30m', '1h', '2h', '6h', '1d'],              # Multiple timeframes
        start_date="2025-01-01",
        end_date="2025-01-02",
        verbosity=Verbosity.PROGRESS
    )

    for symbol, timeframes in results.items():
      for timeframe, df in timeframes.items():
          client.save(df, symbol, timeframe)

if __name__ == '__main__':
    asyncio.run(main())
