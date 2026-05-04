"""
First Bar Breakout Strategy

This strategy trades breakouts from the first bar of the trading day:
- First bar: 9:30 AM ET (first 1-minute bar)
- Entry: Price breaks above first bar high (long) or below first bar low (short)
- Stop Loss: First bar low (long) / First bar high (short)
- Take Profit: Entry price ± first bar range size
- Max 1 trade per day (whichever breakout happens first)
- Exit at 4:00 PM if TP/SL not hit
"""

from dataclasses import dataclass
from datetime import time
from typing import Literal

import polars as pl
from loguru import logger


@dataclass
class TradeResult:
    """Result of a single trade execution."""

    # Trade identification
    date: str  # Date in YYYY-MM-DD format
    direction: Literal["long", "short"]

    # First bar info
    first_bar_time: str  # DateTime of first bar
    first_bar_high: float
    first_bar_low: float
    first_bar_range: float

    # Entry details
    breakout_time: str  # DateTime when breakout occurred
    entry_price: float

    # Exit levels
    stop_loss: float
    take_profit: float

    # Exit details
    exit_time: str  # DateTime when trade exited
    exit_price: float
    exit_reason: Literal["take_profit", "stop_loss", "end_of_day"]

    # Performance
    pnl: float  # Profit/Loss in dollars
    pnl_percent: float  # Profit/Loss as percentage
    bars_held: int  # Number of bars from entry to exit

    def __repr__(self) -> str:
        return (
            f"TradeResult(date={self.date}, {self.direction.upper()}, "
            f"entry={self.entry_price:.2f} @ {self.breakout_time}, "
            f"exit={self.exit_price:.2f} @ {self.exit_time}, "
            f"reason={self.exit_reason}, pnl=${self.pnl:.2f} ({self.pnl_percent:.2%}))"
        )


class FirstBarBreakoutStrategy:
    """
    First Bar Breakout Strategy implementation.

    This strategy identifies the first bar of each trading day and trades
    breakouts above/below that bar's high/low.
    """

    def __init__(
        self,
        first_bar_time: time = time(9, 30),
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
    ):
        """
        Initialize the strategy.

        Args:
            first_bar_time: Time of the first bar (default: 9:30 AM)
            market_open: Market open time (default: 9:30 AM)
            market_close: Market close time (default: 4:00 PM)
        """
        self.first_bar_time = first_bar_time
        self.market_open = market_open
        self.market_close = market_close

    def identify_first_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Identify the first bar (9:30 AM) for each trading day.

        Args:
            df: DataFrame with OHLC data including DateTime column

        Returns:
            DataFrame with first bar high/low for each date
        """
        # Ensure we have date column
        if "date" not in df.columns:
            df = df.with_columns(pl.col("DateTime").dt.date().alias("date"))

        # Filter to first bar time (9:30 AM)
        first_bars = df.filter(
            pl.col("DateTime").dt.time()
            == pl.time(self.first_bar_time.hour, self.first_bar_time.minute)
        )

        # Get first bar high, low, and range for each date
        first_bar_stats = first_bars.select(
            [
                pl.col("date"),
                pl.col("DateTime").alias("first_bar_time"),
                pl.col("High").alias("first_bar_high"),
                pl.col("Low").alias("first_bar_low"),
                (pl.col("High") - pl.col("Low")).alias("first_bar_range"),
            ]
        )

        logger.info(f"Identified first bars for {len(first_bar_stats)} trading days")
        return first_bar_stats

    def detect_breakout(
        self,
        df: pl.DataFrame,
        date: pl.Date,
        first_bar_high: float,
        first_bar_low: float,
    ) -> tuple[Literal["long", "short", "none"], str | None, float | None]:
        """
        Detect the first breakout (high or low) for a given date.

        Args:
            df: Full DataFrame with OHLC data
            date: Date to check for breakout (pl.Date object)
            first_bar_high: First bar high price
            first_bar_low: First bar low price

        Returns:
            Tuple of (direction, breakout_time, breakout_price)
            - direction: "long", "short", or "none"
            - breakout_time: DateTime string when breakout occurred (or None)
            - breakout_price: Price at breakout (or None)
        """
        # Filter data for this date after first bar
        date_data = df.filter(
            (pl.col("date") == pl.lit(date))
            & (
                pl.col("DateTime").dt.time()
                > pl.time(self.first_bar_time.hour, self.first_bar_time.minute)
            )
            & (
                pl.col("DateTime").dt.time()
                <= pl.time(self.market_close.hour, self.market_close.minute)
            )
        ).sort("DateTime")

        if len(date_data) == 0:
            return "none", None, None

        # Check each bar for breakout
        for row in date_data.iter_rows(named=True):
            # Check for high breakout (long)
            if row["High"] > first_bar_high:
                return "long", str(row["DateTime"]), first_bar_high

            # Check for low breakout (short)
            if row["Low"] < first_bar_low:
                return "short", str(row["DateTime"]), first_bar_low

        return "none", None, None

    def execute_trade(
        self,
        df: pl.DataFrame,
        date: pl.Date,
        direction: Literal["long", "short"],
        entry_time: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[str, float, Literal["take_profit", "stop_loss", "end_of_day"], int]:
        """
        Simulate trade execution with stop-loss and take-profit.

        Args:
            df: Full DataFrame with OHLC data
            date: Trade date (pl.Date object)
            direction: "long" or "short"
            entry_time: Entry DateTime string
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price

        Returns:
            Tuple of (exit_time, exit_price, exit_reason, bars_held)
        """
        # Filter data after entry time
        # Convert entry_time string to datetime and replace timezone to match DataFrame
        from datetime import datetime

        entry_dt_obj = datetime.fromisoformat(entry_time)

        trade_data = df.filter(
            (pl.col("date") == pl.lit(date))
            & (pl.col("DateTime").cast(pl.Utf8) > entry_time)
            & (
                pl.col("DateTime").dt.time()
                <= pl.time(self.market_close.hour, self.market_close.minute)
            )
        ).sort("DateTime")

        if len(trade_data) == 0:
            # Exit at entry if no data after entry
            return entry_time, entry_price, "end_of_day", 0

        # Check each bar for exit conditions
        for i, row in enumerate(trade_data.iter_rows(named=True)):
            if direction == "long":
                # Check stop-loss (price goes down)
                if row["Low"] <= stop_loss:
                    return str(row["DateTime"]), stop_loss, "stop_loss", i + 1

                # Check take-profit (price goes up)
                if row["High"] >= take_profit:
                    return str(row["DateTime"]), take_profit, "take_profit", i + 1

            else:  # short
                # Check stop-loss (price goes up)
                if row["High"] >= stop_loss:
                    return str(row["DateTime"]), stop_loss, "stop_loss", i + 1

                # Check take-profit (price goes down)
                if row["Low"] <= take_profit:
                    return str(row["DateTime"]), take_profit, "take_profit", i + 1

        # No exit condition met - exit at end of day
        final_bar = trade_data.row(-1, named=True)
        exit_time = str(final_bar["DateTime"])
        exit_price = float(final_bar["Close"])
        bars_held = len(trade_data)

        return exit_time, exit_price, "end_of_day", bars_held

    def backtest(self, df: pl.DataFrame) -> list[TradeResult]:
        """
        Run the complete backtest on historical data.

        Args:
            df: DataFrame with OHLC data including DateTime column

        Returns:
            List of TradeResult objects
        """
        logger.info("=" * 80)
        logger.info("FIRST BAR BREAKOUT STRATEGY - BACKTEST")
        logger.info("=" * 80)

        # Ensure we have date column
        if "date" not in df.columns:
            df = df.with_columns(pl.col("DateTime").dt.date().alias("date"))

        # Identify first bars for all dates
        first_bars = self.identify_first_bars(df)

        results = []

        # Process each trading day
        for row in first_bars.iter_rows(named=True):
            date = row["date"]  # Keep as pl.Date object
            date_str = str(date)  # String version for display
            first_bar_time = str(row["first_bar_time"])
            first_bar_high = row["first_bar_high"]
            first_bar_low = row["first_bar_low"]
            first_bar_range = row["first_bar_range"]

            # Detect breakout
            direction, breakout_time, entry_price = self.detect_breakout(
                df, date, first_bar_high, first_bar_low
            )

            if direction == "none" or breakout_time is None or entry_price is None:
                logger.debug(f"{date_str}: No breakout detected")
                continue

            # Set stop-loss and take-profit based on direction
            if direction == "long":
                stop_loss = first_bar_low
                take_profit = entry_price + first_bar_range
            else:  # short
                stop_loss = first_bar_high
                take_profit = entry_price - first_bar_range

            # Execute trade
            exit_time, exit_price, exit_reason, bars_held = self.execute_trade(
                df, date, direction, breakout_time, entry_price, stop_loss, take_profit
            )

            # Calculate P&L
            if direction == "long":
                pnl = exit_price - entry_price
            else:  # short
                pnl = entry_price - exit_price

            pnl_percent = pnl / entry_price

            # Create trade result
            trade = TradeResult(
                date=date_str,
                direction=direction,
                first_bar_time=first_bar_time,
                first_bar_high=first_bar_high,
                first_bar_low=first_bar_low,
                first_bar_range=first_bar_range,
                breakout_time=breakout_time,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                pnl_percent=pnl_percent,
                bars_held=bars_held,
            )

            results.append(trade)
            logger.info(f"{trade}")

        logger.info("=" * 80)
        logger.info(f"Total trades executed: {len(results)}")
        logger.info("=" * 80)

        return results

    def generate_summary_stats(self, results: list[TradeResult]) -> dict:
        """
        Generate summary statistics from backtest results.

        Args:
            results: List of TradeResult objects

        Returns:
            Dictionary with performance metrics
        """
        if not results:
            logger.warning("No trades to analyze")
            return {}

        # Separate wins and losses
        wins = [r for r in results if r.pnl > 0]
        losses = [r for r in results if r.pnl < 0]
        breakeven = [r for r in results if r.pnl == 0]

        # Calculate metrics
        total_trades = len(results)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        total_pnl = sum(r.pnl for r in results)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        avg_win = sum(r.pnl for r in wins) / len(wins) if wins else 0
        avg_loss = sum(r.pnl for r in losses) / len(losses) if losses else 0

        gross_profit = sum(r.pnl for r in wins)
        gross_loss = abs(sum(r.pnl for r in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Exit reason breakdown
        tp_exits = len([r for r in results if r.exit_reason == "take_profit"])
        sl_exits = len([r for r in results if r.exit_reason == "stop_loss"])
        eod_exits = len([r for r in results if r.exit_reason == "end_of_day"])

        # Direction breakdown
        long_trades = [r for r in results if r.direction == "long"]
        short_trades = [r for r in results if r.direction == "short"]

        stats = {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "breakeven": len(breakeven),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "tp_exits": tp_exits,
            "sl_exits": sl_exits,
            "eod_exits": eod_exits,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_win_rate": len([r for r in long_trades if r.pnl > 0])
            / len(long_trades)
            if long_trades
            else 0,
            "short_win_rate": len([r for r in short_trades if r.pnl > 0])
            / len(short_trades)
            if short_trades
            else 0,
            "avg_bars_held": sum(r.bars_held for r in results) / total_trades
            if total_trades > 0
            else 0,
        }

        return stats

    def print_summary(self, stats: dict) -> None:
        """
        Print formatted summary statistics.

        Args:
            stats: Dictionary with performance metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(
            f"Wins: {stats['wins']} | Losses: {stats['losses']} | Breakeven: {stats['breakeven']}"
        )
        logger.info(f"Win Rate: {stats['win_rate']:.2%}")
        logger.info(f"Average Win: ${stats['avg_win']:.2f}")
        logger.info(f"Average Loss: ${stats['avg_loss']:.2f}")
        logger.info(f"Average PnL per Trade: ${stats['avg_pnl']:.2f}")
        logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
        logger.info(f"Gross Profit: ${stats['gross_profit']:.2f}")
        logger.info(f"Gross Loss: ${stats['gross_loss']:.2f}")
        logger.info("\nExit Breakdown:")
        logger.info(
            f"  Take Profit: {stats['tp_exits']} ({stats['tp_exits'] / stats['total_trades']:.1%})"
        )
        logger.info(
            f"  Stop Loss: {stats['sl_exits']} ({stats['sl_exits'] / stats['total_trades']:.1%})"
        )
        logger.info(
            f"  End of Day: {stats['eod_exits']} ({stats['eod_exits'] / stats['total_trades']:.1%})"
        )
        logger.info("\nDirection Breakdown:")
        logger.info(
            f"  Long: {stats['long_trades']} trades, {stats['long_win_rate']:.2%} win rate"
        )
        logger.info(
            f"  Short: {stats['short_trades']} trades, {stats['short_win_rate']:.2%} win rate"
        )
        logger.info(f"\nAverage Bars Held: {stats['avg_bars_held']:.1f}")
        logger.info("=" * 80)


def export_results_to_csv(results: list[TradeResult], output_path: str) -> None:
    """
    Export trade results to CSV file.

    Args:
        results: List of TradeResult objects
        output_path: Path to output CSV file
    """
    if not results:
        logger.warning("No results to export")
        return

    # Convert results to dictionary format
    data = {
        "date": [r.date for r in results],
        "direction": [r.direction for r in results],
        "first_bar_time": [r.first_bar_time for r in results],
        "first_bar_high": [r.first_bar_high for r in results],
        "first_bar_low": [r.first_bar_low for r in results],
        "first_bar_range": [r.first_bar_range for r in results],
        "breakout_time": [r.breakout_time for r in results],
        "entry_price": [r.entry_price for r in results],
        "stop_loss": [r.stop_loss for r in results],
        "take_profit": [r.take_profit for r in results],
        "exit_time": [r.exit_time for r in results],
        "exit_price": [r.exit_price for r in results],
        "exit_reason": [r.exit_reason for r in results],
        "pnl": [r.pnl for r in results],
        "pnl_percent": [r.pnl_percent for r in results],
        "bars_held": [r.bars_held for r in results],
    }

    # Create DataFrame and export
    df = pl.DataFrame(data)
    df.write_csv(output_path)
    logger.info(f"Results exported to {output_path}")


# Made with Bob
