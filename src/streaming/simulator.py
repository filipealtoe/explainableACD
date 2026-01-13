"""
StreamingSimulator: Walk-forward temporal processing.

This module simulates real-time streaming by processing tweets in temporal
order, ensuring no data leakage (future data is never used for predictions).

Key features:
- Walk-forward iteration by configurable time windows (1h, 1d)
- Historical context retrieval for baseline statistics
- Memory-efficient: uses Polars lazy frames where possible
- State checkpointing for resumable runs

Usage:
    simulator = StreamingSimulator("data/raw/us_elections_tweets.parquet")
    for timestamp, df_window in simulator.iterate_windows("1h"):
        # Process each hour's tweets
        result = pipeline.process_window(df_window, timestamp)
"""

import logging
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class StreamingSimulator:
    """
    Walk-forward temporal processing with no data leakage.

    This class loads the full dataset once and provides iterators
    that yield data in temporal order, simulating a real-time stream.
    """

    def __init__(
        self,
        data_path: str | Path,
        time_column: str = "created_at",
        text_column: str = "tweet",
        lazy: bool = False,
    ):
        """
        Initialize the simulator.

        Args:
            data_path: Path to parquet file with tweets
            time_column: Column containing timestamps
            text_column: Column containing tweet text
            lazy: If True, use lazy evaluation (memory efficient for large datasets)
        """
        self.data_path = Path(data_path)
        self.time_column = time_column
        self.text_column = text_column

        logger.info(f"Loading dataset from {self.data_path}...")

        # Load and sort by time
        if lazy:
            self._lazy_df = pl.scan_parquet(self.data_path).sort(self.time_column)
            # Get metadata from a small sample
            sample = self._lazy_df.head(1000).collect()
            self._df = None
        else:
            self._df = pl.read_parquet(self.data_path).sort(self.time_column)
            self._lazy_df = None

        # Compute time range
        df = self._df if self._df is not None else self._lazy_df.select(self.time_column).collect()
        self.start_time: datetime = df[self.time_column].min()
        self.end_time: datetime = df[self.time_column].max()
        self.total_rows = len(self._df) if self._df is not None else self._lazy_df.select(pl.len()).collect().item()

        logger.info(f"Loaded {self.total_rows:,} tweets")
        logger.info(f"Time range: {self.start_time} to {self.end_time}")
        logger.info(f"Duration: {(self.end_time - self.start_time).days} days")

    @property
    def df(self) -> pl.DataFrame:
        """Get the full dataframe (loads if lazy)."""
        if self._df is None:
            self._df = self._lazy_df.collect()
        return self._df

    def _parse_window_size(self, window_size: str) -> timedelta:
        """Parse window size string to timedelta."""
        unit = window_size[-1].lower()
        value = int(window_size[:-1])

        if unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "m":
            return timedelta(minutes=value)
        else:
            raise ValueError(f"Unknown window unit: {unit}. Use 'h' (hours), 'd' (days), or 'm' (minutes)")

    def iterate_windows(
        self,
        window_size: str = "1h",
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
    ) -> Generator[tuple[datetime, pl.DataFrame], None, None]:
        """
        Iterate through data in temporal windows.

        Yields (window_start, df_window) tuples in chronological order.
        This is the core method for walk-forward simulation.

        Args:
            window_size: Size of each window ("1h", "6h", "1d")
            start_date: Optional start date (defaults to data start)
            end_date: Optional end date (defaults to data end)

        Yields:
            Tuple of (window_start_timestamp, DataFrame with tweets in that window)
        """
        delta = self._parse_window_size(window_size)

        # Parse dates if strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        start = start_date or self.start_time
        end = end_date or self.end_time

        logger.info(f"Iterating from {start} to {end} with window size {window_size}")

        current = start
        window_count = 0
        total_yielded = 0

        while current < end:
            window_end = current + delta

            # Filter to this window
            df_window = self.df.filter(
                (pl.col(self.time_column) >= current) & (pl.col(self.time_column) < window_end)
            )

            if len(df_window) > 0:
                window_count += 1
                total_yielded += len(df_window)
                yield current, df_window

            current = window_end

        logger.info(f"Iterated {window_count} windows, yielded {total_yielded:,} tweets")

    async def iterate_windows_async(
        self,
        window_size: str = "1h",
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
    ) -> AsyncGenerator[tuple[datetime, pl.DataFrame], None]:
        """
        Async version of iterate_windows for use with async pipelines.

        Same interface as iterate_windows but yields asynchronously.
        """
        for timestamp, df_window in self.iterate_windows(window_size, start_date, end_date):
            yield timestamp, df_window

    def get_historical_context(
        self,
        timestamp: datetime,
        lookback_hours: int = 168,  # 7 days
    ) -> pl.DataFrame:
        """
        Get historical data for baseline statistics.

        CRITICAL: Only returns data BEFORE the given timestamp to prevent
        data leakage. This is used for computing rolling statistics.

        Args:
            timestamp: Current time (exclusive upper bound)
            lookback_hours: How far back to look (default 7 days = 168 hours)

        Returns:
            DataFrame with historical tweets (strictly before timestamp)
        """
        lookback_start = timestamp - timedelta(hours=lookback_hours)

        return self.df.filter(
            (pl.col(self.time_column) >= lookback_start) & (pl.col(self.time_column) < timestamp)
        )

    def get_future_horizon(
        self,
        timestamp: datetime,
        horizon_hours: int = 168,  # 7 days
    ) -> pl.DataFrame:
        """
        Get future data for computing ground truth labels.

        NOTE: This should ONLY be used for offline evaluation, never
        for making predictions in a streaming context.

        Args:
            timestamp: Start time (inclusive)
            horizon_hours: How far forward to look (default 7 days)

        Returns:
            DataFrame with future tweets
        """
        horizon_end = timestamp + timedelta(hours=horizon_hours)

        return self.df.filter(
            (pl.col(self.time_column) >= timestamp) & (pl.col(self.time_column) < horizon_end)
        )

    def get_window_counts(self, window_size: str = "1h") -> pl.DataFrame:
        """
        Get tweet counts per window (useful for visualization).

        Returns DataFrame with columns: [window_start, count]
        """
        # Truncate to window
        if window_size.endswith("h"):
            truncate = f"{window_size[:-1]}h"
        elif window_size.endswith("d"):
            truncate = f"{window_size[:-1]}d"
        else:
            truncate = "1h"

        return (
            self.df.with_columns(pl.col(self.time_column).dt.truncate(truncate).alias("window_start"))
            .group_by("window_start")
            .agg(pl.len().alias("count"))
            .sort("window_start")
        )

    def get_date_range_stats(self) -> dict:
        """Get statistics about the dataset's temporal distribution."""
        counts = self.get_window_counts("1d")

        return {
            "start_date": self.start_time.isoformat(),
            "end_date": self.end_time.isoformat(),
            "total_days": (self.end_time - self.start_time).days,
            "total_tweets": self.total_rows,
            "avg_tweets_per_day": self.total_rows / max(1, (self.end_time - self.start_time).days),
            "min_tweets_per_day": counts["count"].min(),
            "max_tweets_per_day": counts["count"].max(),
            "days_with_data": len(counts),
        }


def create_train_test_split(
    simulator: StreamingSimulator,
    test_start_date: datetime | str,
    gap_hours: int = 24,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Create temporal train/test split with gap to prevent leakage.

    Args:
        simulator: StreamingSimulator instance
        test_start_date: Start of test period
        gap_hours: Hours between train end and test start (prevents leakage)

    Returns:
        Tuple of (train_df, test_df)
    """
    if isinstance(test_start_date, str):
        test_start_date = datetime.fromisoformat(test_start_date)

    train_end = test_start_date - timedelta(hours=gap_hours)

    train_df = simulator.df.filter(pl.col(simulator.time_column) < train_end)
    test_df = simulator.df.filter(pl.col(simulator.time_column) >= test_start_date)

    logger.info(f"Train: {len(train_df):,} tweets (until {train_end})")
    logger.info(f"Test: {len(test_df):,} tweets (from {test_start_date})")

    return train_df, test_df
