#!/usr/bin/env python
"""
Visualize Temporal Distribution of Tweet Dataset

Creates visualizations to understand the data collection pattern.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))


def load_and_prepare_data(input_path: Path) -> pl.DataFrame:
    """Load data and add temporal columns."""
    df = pl.read_parquet(input_path)
    df = df.sort("created_at_utc")

    df = df.with_columns(
        [
            pl.col("created_at_utc").dt.truncate("1h").alias("hour_bucket"),
            pl.col("created_at_utc").dt.truncate("1d").alias("day_bucket"),
            pl.col("created_at_utc").dt.year().alias("year"),
            pl.col("created_at_utc").dt.month().alias("month"),
            pl.col("created_at_utc").dt.day().alias("day_of_month"),
            pl.col("created_at_utc").dt.hour().alias("hour_of_day"),
        ]
    )
    return df


def plot_collection_timeline(df: pl.DataFrame, output_dir: Path):
    """
    Plot 1: Timeline showing when data was collected.
    Shows gaps clearly.
    """
    daily = df.group_by("day_bucket").agg(pl.len().alias("count")).sort("day_bucket")

    fig, ax = plt.subplots(figsize=(16, 4))

    dates = daily["day_bucket"].to_list()
    counts = daily["count"].to_numpy()

    ax.bar(dates, counts, width=1, color="steelblue", alpha=0.8)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Tweets per Day", fontsize=12)
    ax.set_title("Data Collection Timeline: Tweets per Day (Gaps = No Data Collected)", fontsize=14)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "01_collection_timeline.png", dpi=150)
    plt.close()
    print("Saved: 01_collection_timeline.png")


def plot_day_of_month_pattern(df: pl.DataFrame, output_dir: Path):
    """
    Plot 2: Which days of the month have data?
    Shows the end-of-month collection pattern.
    """
    # Count unique collection dates per day-of-month
    daily = df.group_by("day_bucket").agg(pl.len().alias("count"))
    daily = daily.with_columns(pl.col("day_bucket").dt.day().alias("day_of_month"))

    # Count how many times each day-of-month appears
    dom_counts = (
        daily.group_by("day_of_month")
        .agg(
            pl.len().alias("n_occurrences"),
            pl.col("count").sum().alias("total_tweets"),
        )
        .sort("day_of_month")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    days = dom_counts["day_of_month"].to_numpy()
    occurrences = dom_counts["n_occurrences"].to_numpy()
    tweets = dom_counts["total_tweets"].to_numpy()

    # Left: Number of collection days per day-of-month
    colors = ["#d62728" if d >= 26 else "steelblue" for d in days]
    ax1.bar(days, occurrences, color=colors, alpha=0.8)
    ax1.set_xlabel("Day of Month", fontsize=12)
    ax1.set_ylabel("Number of Collection Days", fontsize=12)
    ax1.set_title("Data Collection Pattern by Day of Month\n(Red = Days 26-31)", fontsize=13)
    ax1.set_xticks(range(1, 32))
    ax1.grid(axis="y", alpha=0.3)

    # Add annotation
    total_dom_26_31 = occurrences[days >= 26].sum()
    total_all = occurrences.sum()
    pct = 100 * total_dom_26_31 / total_all
    ax1.annotate(
        f"Days 26-31: {total_dom_26_31}/{total_all} ({pct:.0f}%)",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Right: Total tweets per day-of-month
    ax2.bar(days, tweets / 1000, color=colors, alpha=0.8)
    ax2.set_xlabel("Day of Month", fontsize=12)
    ax2.set_ylabel("Total Tweets (thousands)", fontsize=12)
    ax2.set_title("Total Tweets by Day of Month", fontsize=13)
    ax2.set_xticks(range(1, 32))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "02_day_of_month_pattern.png", dpi=150)
    plt.close()
    print("Saved: 02_day_of_month_pattern.png")


def plot_collection_windows(df: pl.DataFrame, output_dir: Path):
    """
    Plot 3: Show the actual collection windows (consecutive days with data).
    """
    daily = df.group_by("day_bucket").agg(pl.len().alias("count")).sort("day_bucket")
    days_with_data = daily["day_bucket"].to_list()

    # Find consecutive windows
    windows = []
    window_start = days_with_data[0]
    window_end = days_with_data[0]

    for i in range(1, len(days_with_data)):
        diff = (days_with_data[i] - days_with_data[i - 1]).days
        if diff == 1:
            window_end = days_with_data[i]
        else:
            windows.append((window_start, window_end))
            window_start = days_with_data[i]
            window_end = days_with_data[i]
    windows.append((window_start, window_end))

    # Calculate window lengths
    window_lengths = [(end - start).days + 1 for start, end in windows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Top: Timeline with windows highlighted
    ax1.set_xlim(days_with_data[0], days_with_data[-1])
    ax1.set_ylim(0, 1)

    for start, end in windows:
        ax1.axvspan(start, end, alpha=0.6, color="steelblue")

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title(f"Data Collection Windows ({len(windows)} windows over {len(days_with_data)} days)", fontsize=14)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_yticks([])
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Bottom: Histogram of window lengths
    ax2.hist(
        window_lengths,
        bins=range(1, max(window_lengths) + 2),
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        align="left",
    )
    ax2.set_xlabel("Window Length (consecutive days)", fontsize=12)
    ax2.set_ylabel("Number of Windows", fontsize=12)
    ax2.set_title(
        f"Distribution of Collection Window Lengths\n(Mean: {np.mean(window_lengths):.1f} days, Median: {np.median(window_lengths):.0f} days)",
        fontsize=13,
    )
    ax2.set_xticks(range(1, max(window_lengths) + 1))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "03_collection_windows.png", dpi=150)
    plt.close()
    print("Saved: 03_collection_windows.png")

    return windows, window_lengths


def plot_hourly_distribution(df: pl.DataFrame, output_dir: Path):
    """
    Plot 4: Tweets per hour distribution.
    """
    hourly = df.group_by("hour_bucket").agg(pl.len().alias("count"))
    counts = hourly["count"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of tweets per hour
    ax1.hist(counts, bins=50, color="steelblue", alpha=0.8, edgecolor="black")
    ax1.axvline(np.mean(counts), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(counts):.0f}")
    ax1.axvline(
        np.median(counts), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(counts):.0f}"
    )
    ax1.set_xlabel("Tweets per Hour", fontsize=12)
    ax1.set_ylabel("Number of Hours", fontsize=12)
    ax1.set_title("Distribution of Tweets per Hour\n(Only hours with tweets)", fontsize=13)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: By hour of day
    hour_of_day = df.group_by("hour_of_day").agg(pl.len().alias("count")).sort("hour_of_day")
    hours = hour_of_day["hour_of_day"].to_numpy()
    hour_counts = hour_of_day["count"].to_numpy()

    ax2.bar(hours, hour_counts / 1000, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Hour of Day (UTC)", fontsize=12)
    ax2.set_ylabel("Total Tweets (thousands)", fontsize=12)
    ax2.set_title("Tweets by Hour of Day (UTC)", fontsize=13)
    ax2.set_xticks(range(0, 24))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "04_hourly_distribution.png", dpi=150)
    plt.close()
    print("Saved: 04_hourly_distribution.png")


def plot_yearly_monthly(df: pl.DataFrame, output_dir: Path):
    """
    Plot 5: Tweets by year and month.
    """
    # By year
    yearly = df.group_by("year").agg(pl.len().alias("count")).sort("year")

    # By year-month
    df = df.with_columns(
        (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias("year_month")
    )
    monthly = df.group_by("year_month").agg(pl.len().alias("count")).sort("year_month")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Top: By year
    years = yearly["year"].to_numpy()
    year_counts = yearly["count"].to_numpy()
    bars = ax1.bar(years, year_counts / 1000, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Tweets (thousands)", fontsize=12)
    ax1.set_title("Tweets by Year", fontsize=14)
    ax1.set_xticks(years)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, count in zip(bars, year_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{count:,}", ha="center", va="bottom", fontsize=10
        )

    # Bottom: By month
    months = monthly["year_month"].to_list()
    month_counts = monthly["count"].to_numpy()
    ax2.bar(range(len(months)), month_counts / 1000, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Year-Month", fontsize=12)
    ax2.set_ylabel("Tweets (thousands)", fontsize=12)
    ax2.set_title("Tweets by Month", fontsize=14)
    ax2.set_xticks(range(0, len(months), 3))
    ax2.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "05_yearly_monthly.png", dpi=150)
    plt.close()
    print("Saved: 05_yearly_monthly.png")


def plot_summary_stats(df: pl.DataFrame, output_dir: Path, windows: list, window_lengths: list):
    """
    Plot 6: Summary statistics infographic.
    """
    # Calculate stats
    total_tweets = len(df)
    date_range = f"{df['created_at_utc'].min().date()} to {df['created_at_utc'].max().date()}"

    hourly = df.group_by("hour_bucket").agg(pl.len().alias("count"))
    daily = df.group_by("day_bucket").agg(pl.len().alias("count"))

    hours_with_data = len(hourly)
    days_with_data = len(daily)

    min_hour = hourly["hour_bucket"].min()
    max_hour = hourly["hour_bucket"].max()
    total_hours = int((max_hour - min_hour).total_seconds() / 3600) + 1

    min_day = daily["day_bucket"].min()
    max_day = daily["day_bucket"].max()
    total_days = (max_day - min_day).days + 1

    hourly_counts = hourly["count"].to_numpy()
    daily_counts = daily["count"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.95,
        "AI Tweets Dataset - Temporal Summary",
        fontsize=20,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )

    # Stats text
    stats_text = f"""
    DATASET OVERVIEW
    ─────────────────────────────────────
    Total Tweets:          {total_tweets:,}
    Date Range:            {date_range}

    COVERAGE (Sparse Data!)
    ─────────────────────────────────────
    Hours with tweets:     {hours_with_data:,} / {total_hours:,} ({100 * hours_with_data / total_hours:.1f}%)
    Days with tweets:      {days_with_data:,} / {total_days:,} ({100 * days_with_data / total_days:.1f}%)

    DATA COLLECTION PATTERN
    ─────────────────────────────────────
    Collection windows:    {len(windows)} batches
    Window length:         {np.mean(window_lengths):.1f} days avg (median: {np.median(window_lengths):.0f})
    Pattern:               ~End of month (days 26-31)

    TWEETS PER HOUR (when active)
    ─────────────────────────────────────
    Mean:                  {hourly_counts.mean():.0f}
    Median:                {np.median(hourly_counts):.0f}
    Min:                   {hourly_counts.min()}
    Max:                   {hourly_counts.max():,}
    P25-P75:               {np.percentile(hourly_counts, 25):.0f} - {np.percentile(hourly_counts, 75):.0f}

    TWEETS PER DAY (when active)
    ─────────────────────────────────────
    Mean:                  {daily_counts.mean():,.0f}
    Median:                {np.median(daily_counts):,.0f}
    Min:                   {daily_counts.min():,}
    Max:                   {daily_counts.max():,}
    P25-P75:               {np.percentile(daily_counts, 25):,.0f} - {np.percentile(daily_counts, 75):,.0f}

    KEY INSIGHT
    ─────────────────────────────────────
    This is NOT continuous streaming data!
    Data was collected in periodic batches,
    typically ~5 consecutive days per month,
    concentrated at the end of each month.
    """

    ax.text(0.1, 0.85, stats_text, fontsize=12, fontfamily="monospace", verticalalignment="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / "00_summary_stats.png", dpi=150)
    plt.close()
    print("Saved: 00_summary_stats.png")


def main():
    input_path = repo_root / "data" / "processed" / "tweets_v9.parquet"
    output_dir = repo_root / "experiments" / "visualizations" / "temporal"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CREATING TEMPORAL DISTRIBUTION VISUALIZATIONS")
    print("=" * 60)

    df = load_and_prepare_data(input_path)
    print(f"Loaded {len(df):,} tweets")

    print("\nGenerating plots...")
    plot_collection_timeline(df, output_dir)
    plot_day_of_month_pattern(df, output_dir)
    windows, window_lengths = plot_collection_windows(df, output_dir)
    plot_hourly_distribution(df, output_dir)
    plot_yearly_monthly(df, output_dir)
    plot_summary_stats(df, output_dir, windows, window_lengths)

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
