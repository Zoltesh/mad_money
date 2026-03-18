"""Fetch orchestration engine for OHLCV retrieval."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import polars as pl
from rich.progress import Progress

from src.data.progress import build_shared_progress


def _mark_task_failed(shared_progress: Progress | None, task_id: Any, label: str) -> None:
    """Mark a progress task as failed."""
    if shared_progress is not None and task_id is not None:
        shared_progress.update(task_id, completed=0)
        shared_progress.update(task_id, description=f"[red]{label} (failed)[/red]")


def _build_bounded_batch_starts(
    start_ts: int, end_ts: int, timeframe_ms: int, coinbase_candle_limit: int
) -> list[int]:
    """Build deterministic batch start timestamps for bounded fetch windows."""
    if start_ts > end_ts:
        return []
    step_ms = timeframe_ms * coinbase_candle_limit
    return list(range(start_ts, end_ts + 1, step_ms))


async def _fetch_bounded_window(
    *,
    client: Any,
    exchange: Any,
    semaphore: Any,
    symbol: str,
    timeframe: str,
    planned_since: int,
    window_end: int,
    end_ts: int,
    timeframe_ms: int,
    shared_progress: Progress | None,
    activity_state: dict[str, Any] | None,
) -> list[list[float]]:
    """Fetch all candles for a bounded window start."""
    cursor = planned_since
    window_candles: list[list[float]] = []

    while cursor <= window_end:
        client._update_activity_progress(shared_progress, activity_state, active_delta=1)
        try:
            candles = await client._fetch_batch(
                exchange, semaphore, symbol, timeframe, int(cursor)
            )
        except Exception:
            client._update_activity_progress(
                shared_progress, activity_state, active_delta=-1
            )
            raise
        client._update_activity_progress(shared_progress, activity_state, active_delta=-1)

        if not candles:
            break

        window_candles.extend([c for c in candles if c[0] <= end_ts and c[0] <= window_end])
        last_candle_ts = candles[-1][0]
        if last_candle_ts < cursor:
            break

        next_cursor = max(last_candle_ts + timeframe_ms, cursor + timeframe_ms)
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    return window_candles


async def _execute_fetch_sequential(
    *,
    client: Any,
    exchange: Any,
    semaphore: Any,
    symbol: str,
    timeframe: str,
    since: int | None,
    end_ts: int,
    end_date: str | None,
    timeframe_ms: int,
    collect_results: bool,
    on_batch: Any,
    progress_task_id: Any,
    shared_progress: Progress | None,
    activity_state: dict[str, Any] | None,
    progress_tracker: Any,
    use_shared_progress: bool,
    expected_candles: int,
    coinbase_candle_limit: int,
    max_consecutive_retryable_failures_no_end_date: int,
    retryable_exceptions: tuple[type[BaseException], ...],
    non_retryable_exceptions: tuple[type[BaseException], ...],
    logger: Any,
) -> tuple[list[list[float]], int, int]:
    """Execute legacy sequential batch progression for one combo."""
    all_candles: list[list[float]] = []
    pending_advance = 0
    failed_batches = 0
    failed_batch_starts: list[int] = []
    consecutive_failed_batches = 0
    total_candles_fetched = 0

    while since is not None:
        if end_ts and since > end_ts:
            break

        try:
            client._update_activity_progress(
                shared_progress, activity_state, active_delta=1
            )
            candles = await client._fetch_batch(
                exchange, semaphore, symbol, timeframe, int(since)
            )
            client._update_activity_progress(
                shared_progress, activity_state, active_delta=-1
            )
        except non_retryable_exceptions:
            client._update_activity_progress(
                shared_progress, activity_state, active_delta=-1
            )
            raise
        except retryable_exceptions as e:
            client._update_activity_progress(
                shared_progress,
                activity_state,
                active_delta=-1,
                failed_increment=1,
            )
            logger.warning(
                "Batch failed after retries for %s %s (timestamp %d): %s - %s. Continuing with remaining batches.",
                symbol,
                timeframe,
                since,
                type(e).__name__,
                e,
            )
            failed_batches += 1
            failed_batch_starts.append(since)
            consecutive_failed_batches += 1

            if (
                end_date is None
                and consecutive_failed_batches
                >= max_consecutive_retryable_failures_no_end_date
            ):
                raise

            since = since + (timeframe_ms * coinbase_candle_limit)
            pending_advance = client._update_progress(
                pending_advance,
                progress_task_id,
                shared_progress,
                progress_tracker,
                use_shared_progress,
                total_candles_fetched,
                expected_candles,
                activity_state,
            )
            continue

        if not candles:
            client._flush_progress(
                pending_advance,
                progress_task_id,
                shared_progress,
                progress_tracker,
                use_shared_progress,
                extra=1,
                activity_state=activity_state,
            )
            break

        consecutive_failed_batches = 0
        last_candle_ts = candles[-1][0]
        batch_candles = candles

        if end_ts is not None:
            batch_candles = [c for c in batch_candles if c[0] <= end_ts]

        if batch_candles:
            total_candles_fetched += len(batch_candles)
            if collect_results:
                all_candles.extend(batch_candles)
            if on_batch is not None:
                await on_batch(client._candles_to_dataframe(batch_candles))

        pending_advance = client._update_progress(
            pending_advance,
            progress_task_id,
            shared_progress,
            progress_tracker,
            use_shared_progress,
            total_candles_fetched,
            expected_candles,
            activity_state,
        )

        if end_ts and last_candle_ts >= end_ts:
            client._flush_progress(
                pending_advance,
                progress_task_id,
                shared_progress,
                progress_tracker,
                use_shared_progress,
                activity_state=activity_state,
            )
            since = None
        else:
            next_since = last_candle_ts + timeframe_ms
            if next_since <= since:
                next_since = since + timeframe_ms

            if last_candle_ts < since:
                client._flush_progress(
                    pending_advance,
                    progress_task_id,
                    shared_progress,
                    progress_tracker,
                    use_shared_progress,
                    activity_state=activity_state,
                )
                since = None
            else:
                since = next_since

    if failed_batch_starts and end_date is not None:
        for failed_since in sorted(set(failed_batch_starts)):
            recovered = False
            for retry_round in range(client.max_retries + 1):
                try:
                    client._update_activity_progress(
                        shared_progress, activity_state, active_delta=1
                    )
                    retry_candles = await client._fetch_batch(
                        exchange, semaphore, symbol, timeframe, int(failed_since)
                    )
                    client._update_activity_progress(
                        shared_progress, activity_state, active_delta=-1
                    )
                    recovered = True
                    if retry_candles:
                        batch_candles = [c for c in retry_candles if c[0] <= end_ts]
                        if batch_candles:
                            total_candles_fetched += len(batch_candles)
                            if collect_results:
                                all_candles.extend(batch_candles)
                            if on_batch is not None:
                                await on_batch(client._candles_to_dataframe(batch_candles))
                    break
                except non_retryable_exceptions:
                    client._update_activity_progress(
                        shared_progress, activity_state, active_delta=-1
                    )
                    raise
                except retryable_exceptions as e:
                    client._update_activity_progress(
                        shared_progress, activity_state, active_delta=-1
                    )
                    if retry_round < client.max_retries:
                        logger.warning(
                            "Gap refill retry %d/%d for %s %s (timestamp %d) after %s - %s",
                            retry_round + 1,
                            client.max_retries + 1,
                            symbol,
                            timeframe,
                            failed_since,
                            type(e).__name__,
                            e,
                        )
                        continue
                    logger.error(
                        "Gap refill failed for %s %s (timestamp %d): %s - %s",
                        symbol,
                        timeframe,
                        failed_since,
                        type(e).__name__,
                        e,
                    )

            if recovered:
                failed_batches = max(0, failed_batches - 1)
                client._update_activity_progress(
                    shared_progress, activity_state, failed_increment=-1
                )

    return all_candles, total_candles_fetched, failed_batches


async def _execute_fetch_concurrent_bounded(
    *,
    client: Any,
    exchange: Any,
    semaphore: Any,
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
    timeframe_ms: int,
    collect_results: bool,
    on_batch: Any,
    progress_task_id: Any,
    shared_progress: Progress | None,
    activity_state: dict[str, Any] | None,
    progress_tracker: Any,
    use_shared_progress: bool,
    expected_candles: int,
    coinbase_candle_limit: int,
    retryable_exceptions: tuple[type[BaseException], ...],
    non_retryable_exceptions: tuple[type[BaseException], ...],
    logger: Any,
) -> tuple[list[list[float]], int, int]:
    """Execute bounded historical fetch with concurrent windows per combo."""
    planned_sinces = _build_bounded_batch_starts(
        start_ts, end_ts, timeframe_ms, coinbase_candle_limit
    )
    if not planned_sinces:
        return [], 0, 0

    step_ms = timeframe_ms * coinbase_candle_limit
    worker_count = min(client.batch_concurrency, len(planned_sinces))
    window_queue: asyncio.Queue[int | None] = asyncio.Queue()
    for planned_since in planned_sinces:
        window_queue.put_nowait(planned_since)
    for _ in range(worker_count):
        window_queue.put_nowait(None)

    all_candles: list[list[float]] = []
    pending_advance = 0
    failed_batches = 0
    total_candles_fetched = 0
    pending_batch_writes: set[asyncio.Task[None]] = set()
    write_limit = max(1, client.batch_queue_size)
    results_queue: asyncio.Queue[
        tuple[str, int, list[list[float]] | None, BaseException | None]
    ] = asyncio.Queue()
    failed_planned_sinces: list[int] = []

    async def _queue_batch_write(batch_candles: list[list[float]]) -> None:
        nonlocal total_candles_fetched, pending_batch_writes
        filtered = [c for c in batch_candles if c[0] <= end_ts]
        if not filtered:
            return
        total_candles_fetched += len(filtered)
        if collect_results:
            all_candles.extend(filtered)
        if on_batch is not None:
            batch_df = client._candles_to_dataframe(filtered)
            task = asyncio.create_task(on_batch(batch_df))
            pending_batch_writes.add(task)
            if len(pending_batch_writes) >= write_limit:
                done, pending = await asyncio.wait(
                    pending_batch_writes,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending_batch_writes.clear()
                pending_batch_writes.update(pending)
                for completed in done:
                    await completed

    async def _fetch_worker() -> None:
        while True:
            planned_since = await window_queue.get()
            if planned_since is None:
                window_queue.task_done()
                break
            try:
                window_end = min(end_ts, planned_since + step_ms - timeframe_ms)
                window_candles = await _fetch_bounded_window(
                    client=client,
                    exchange=exchange,
                    semaphore=semaphore,
                    symbol=symbol,
                    timeframe=timeframe,
                    planned_since=planned_since,
                    window_end=window_end,
                    end_ts=end_ts,
                    timeframe_ms=timeframe_ms,
                    shared_progress=shared_progress,
                    activity_state=activity_state,
                )
                await results_queue.put(("ok", planned_since, window_candles, None))
            except non_retryable_exceptions as e:
                await results_queue.put(("fatal", planned_since, None, e))
            except retryable_exceptions as e:
                client._update_activity_progress(
                    shared_progress,
                    activity_state,
                    failed_increment=1,
                )
                await results_queue.put(("retryable_failed", planned_since, None, e))
            finally:
                window_queue.task_done()

    workers = [asyncio.create_task(_fetch_worker()) for _ in range(worker_count)]
    processed = 0
    fatal_error: BaseException | None = None
    try:
        while processed < len(planned_sinces):
            status, planned_since, candles, exc = await results_queue.get()
            processed += 1

            if status == "fatal":
                fatal_error = exc
                break

            if status == "retryable_failed":
                failed_batches += 1
                failed_planned_sinces.append(planned_since)
                logger.warning(
                    "Batch failed after retries for %s %s (timestamp %d): %s - %s. Continuing with remaining batches.",
                    symbol,
                    timeframe,
                    planned_since,
                    type(exc).__name__,
                    exc,
                )
            elif candles:
                await _queue_batch_write(candles)

            pending_advance = client._update_progress(
                pending_advance,
                progress_task_id,
                shared_progress,
                progress_tracker,
                use_shared_progress,
                total_candles_fetched,
                expected_candles,
                activity_state,
            )

        if failed_planned_sinces:
            for failed_since in sorted(set(failed_planned_sinces)):
                recovered = False
                for retry_round in range(client.max_retries + 1):
                    try:
                        window_end = min(end_ts, failed_since + step_ms - timeframe_ms)
                        retry_candles = await _fetch_bounded_window(
                            client=client,
                            exchange=exchange,
                            semaphore=semaphore,
                            symbol=symbol,
                            timeframe=timeframe,
                            planned_since=failed_since,
                            window_end=window_end,
                            end_ts=end_ts,
                            timeframe_ms=timeframe_ms,
                            shared_progress=shared_progress,
                            activity_state=activity_state,
                        )
                        recovered = True
                        if retry_candles:
                            await _queue_batch_write(retry_candles)
                        break
                    except non_retryable_exceptions:
                        raise
                    except retryable_exceptions as e:
                        if retry_round < client.max_retries:
                            logger.warning(
                                "Gap refill retry %d/%d for %s %s (timestamp %d) after %s - %s",
                                retry_round + 1,
                                client.max_retries + 1,
                                symbol,
                                timeframe,
                                failed_since,
                                type(e).__name__,
                                e,
                            )
                            continue
                        logger.error(
                            "Gap refill failed for %s %s (timestamp %d): %s - %s",
                            symbol,
                            timeframe,
                            failed_since,
                            type(e).__name__,
                            e,
                        )

                if recovered:
                    failed_batches = max(0, failed_batches - 1)
                    client._update_activity_progress(
                        shared_progress, activity_state, failed_increment=-1
                    )

        if pending_batch_writes:
            await asyncio.gather(*pending_batch_writes)

        if fatal_error is not None:
            raise fatal_error
    finally:
        client._flush_progress(
            pending_advance,
            progress_task_id,
            shared_progress,
            progress_tracker,
            use_shared_progress,
            activity_state=activity_state,
        )
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    return all_candles, total_candles_fetched, failed_batches


async def execute_fetch(
    *,
    client: Any,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str | None,
    verbosity: Any,
    progress_task_id: Any,
    shared_progress: Progress | None,
    activity_state: dict[str, Any] | None,
    on_batch: Any,
    collect_results: bool,
    verbosity_disabled: Any,
    verbosity_verbose: Any,
    ohlcv_schema: dict[str, Any],
    timeframe_seconds: dict[str, int],
    calculate_expected_batches: Any,
    coinbase_candle_limit: int,
    max_consecutive_retryable_failures_no_end_date: int,
    retryable_exceptions: tuple[type[BaseException], ...],
    non_retryable_exceptions: tuple[type[BaseException], ...],
    logger: Any,
) -> pl.DataFrame:
    """Run one symbol/timeframe historical fetch loop."""
    effective_verbosity = client._resolve_verbosity(verbosity)
    client._validate_timeframe(timeframe)

    exchange = client._get_exchange()
    semaphore = client._get_semaphore()

    start_ts = int(client._parse_date(start_date).timestamp() * 1000)
    current_ts = datetime.now(UTC).timestamp() * 1000
    if end_date:
        end_ts = int(client._parse_date(end_date, end_of_day=True).timestamp() * 1000)
    else:
        end_ts = int(current_ts)

    if start_ts > end_ts:
        return pl.DataFrame(schema=ohlcv_schema)

    expected_batches = calculate_expected_batches(start_ts, end_ts, timeframe)
    expected_candles = client._calculate_expected_candles(start_ts, end_ts, timeframe)

    use_shared_progress = shared_progress is not None and progress_task_id is not None

    progress_tracker = None
    if effective_verbosity != verbosity_disabled and not use_shared_progress:
        progress_tracker = client._progress_tracker_factory(
            total=expected_batches,
            symbol=symbol,
            timeframe=timeframe,
            verbosity=effective_verbosity,
        )
        progress_tracker.start()

        if effective_verbosity == verbosity_verbose:
            end_display = end_date if end_date else "latest"
            print(
                f"Starting fetch for {symbol} {timeframe} from {start_date} to {end_display}..."
            )

    timeframe_ms = timeframe_seconds[timeframe] * 1000
    all_candles: list[list[float]] = []
    total_candles_fetched = 0
    failed_batches = 0
    use_concurrent_bounded = (
        end_date is not None
        and client.enable_intra_combo_concurrency
        and client.batch_concurrency > 1
    )

    try:
        if use_concurrent_bounded:
            all_candles, total_candles_fetched, failed_batches = (
                await _execute_fetch_concurrent_bounded(
                    client=client,
                    exchange=exchange,
                    semaphore=semaphore,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    timeframe_ms=timeframe_ms,
                    collect_results=collect_results,
                    on_batch=on_batch,
                    progress_task_id=progress_task_id,
                    shared_progress=shared_progress,
                    activity_state=activity_state,
                    progress_tracker=progress_tracker,
                    use_shared_progress=use_shared_progress,
                    expected_candles=expected_candles,
                    coinbase_candle_limit=coinbase_candle_limit,
                    retryable_exceptions=retryable_exceptions,
                    non_retryable_exceptions=non_retryable_exceptions,
                    logger=logger,
                )
            )
        else:
            all_candles, total_candles_fetched, failed_batches = (
                await _execute_fetch_sequential(
                    client=client,
                    exchange=exchange,
                    semaphore=semaphore,
                    symbol=symbol,
                    timeframe=timeframe,
                    since=start_ts,
                    end_ts=end_ts,
                    end_date=end_date,
                    timeframe_ms=timeframe_ms,
                    collect_results=collect_results,
                    on_batch=on_batch,
                    progress_task_id=progress_task_id,
                    shared_progress=shared_progress,
                    activity_state=activity_state,
                    progress_tracker=progress_tracker,
                    use_shared_progress=use_shared_progress,
                    expected_candles=expected_candles,
                    coinbase_candle_limit=coinbase_candle_limit,
                    max_consecutive_retryable_failures_no_end_date=max_consecutive_retryable_failures_no_end_date,
                    retryable_exceptions=retryable_exceptions,
                    non_retryable_exceptions=non_retryable_exceptions,
                    logger=logger,
                )
            )

        if failed_batches > 0:
            logger.warning(
                "Fetch completed for %s %s: %d candles fetched, %d batches failed",
                symbol,
                timeframe,
                total_candles_fetched,
                failed_batches,
            )
    finally:
        if progress_tracker is not None:
            progress_tracker.close()
            if effective_verbosity == verbosity_verbose:
                print(
                    f"Completed fetch for {symbol} {timeframe}: {total_candles_fetched} candles fetched"
                )

    if not collect_results:
        return pl.DataFrame(schema=ohlcv_schema)

    if not all_candles:
        return pl.DataFrame(schema=ohlcv_schema)

    return (
        client._candles_to_dataframe(all_candles)
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )


async def execute_fetch_multiple(
    *,
    client: Any,
    symbols: list[str],
    timeframes: list[str],
    start_date: str,
    end_date: str | None,
    verbosity: Any,
    verbosity_disabled: Any,
    verbosity_verbose: Any,
    calculate_expected_batches: Any,
    logger: Any,
    progress_class: Any,
    get_progress_color: Any,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Run concurrent fetch for all symbol/timeframe combinations."""
    effective_verbosity = client._resolve_verbosity(verbosity)
    combinations = [(s, t) for s in symbols for t in timeframes]

    if effective_verbosity == verbosity_verbose:
        end_display = end_date if end_date else "latest"
        print(
            f"Starting batch fetch for {len(combinations)} symbol/timeframe combinations "
            f"from {start_date} to {end_display}..."
        )

    shared_progress: Progress | None = None
    task_ids: dict[tuple[str, str], Any] = {}
    if effective_verbosity != verbosity_disabled:
        shared_progress = build_shared_progress(progress_class)
        shared_progress.start()

        start_ts = int(client._parse_date(start_date).timestamp() * 1000)
        if end_date:
            end_ts = int(client._parse_date(end_date, end_of_day=True).timestamp() * 1000)
        else:
            end_ts = int(datetime.now(UTC).timestamp() * 1000)

        for symbol, timeframe in combinations:
            color = get_progress_color(symbol, timeframe)
            description = f"[{color}]{symbol} {timeframe}[/{color}]"
            expected_batches = calculate_expected_batches(start_ts, end_ts, timeframe)
            task_id = shared_progress.add_task(description, total=expected_batches)
            task_ids[(symbol, timeframe)] = task_id

    async def fetch_one(
        symbol: str, timeframe: str, task_id: Any
    ) -> tuple[str, str, pl.DataFrame]:
        df = await client.fetch(
            symbol,
            timeframe,
            start_date,
            end_date,
            effective_verbosity,
            progress_task_id=task_id,
            shared_progress=shared_progress,
        )
        return (symbol, timeframe, df)

    tasks = []
    task_metadata = []
    for symbol, timeframe in combinations:
        task_id = task_ids.get((symbol, timeframe))
        tasks.append(fetch_one(symbol, timeframe, task_id))
        task_metadata.append((symbol, timeframe))

    results = await client._gather_with_exceptions(tasks)

    result_dict: dict[str, dict[str, pl.DataFrame]] = {}
    total_candles = 0
    failed_tasks = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            symbol, timeframe = task_metadata[i]
            task_id = task_ids.get((symbol, timeframe))
            _mark_task_failed(shared_progress, task_id, f"{symbol} {timeframe}")
            failed_tasks.append((symbol, timeframe, result))
            continue

        symbol, timeframe, df = result
        if symbol not in result_dict:
            result_dict[symbol] = {}
        result_dict[symbol][timeframe] = df
        total_candles += len(df)

    if failed_tasks:
        for symbol, timeframe, exc in failed_tasks:
            print(
                f"Warning: Failed to fetch {symbol}/{timeframe}: {type(exc).__name__}: {exc}"
            )

    if effective_verbosity == verbosity_verbose:
        print(f"Completed batch fetch: {total_candles} total candles fetched")

    if shared_progress is not None:
        shared_progress.stop()

    return result_dict
