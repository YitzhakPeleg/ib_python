"""Backtesting framework for evaluating trading signals."""

from typing import Literal

import polars as pl
from loguru import logger

from models import SignalResult, SignalType, TradeSetup


def backtest_trade_setups(
    df: pl.DataFrame,
    trade_setups: list[TradeSetup],
    timezone: str = "US/Eastern",
) -> pl.DataFrame:
    """
    Backtest a list of trade setups against actual price data.

    Args:
        df: Full DataFrame with OHLC price data
        trade_setups: List of TradeSetup objects to backtest
        timezone: Timezone for the data

    Returns:
        DataFrame with backtest results for each trade
    """
    results = []

    for setup in trade_setups:
        result = backtest_single_trade(df, setup, timezone)
        if result:
            results.append(result)

    if not results:
        logger.warning("No backtest results generated")
        return pl.DataFrame()

    # Convert results to DataFrame
    results_data = {
        "date": [r.setup.date for r in results],
        "signal": [r.setup.signal.name for r in results],
        "entry_price": [r.setup.entry_price for r in results],
        "stop_loss": [r.setup.stop_loss for r in results],
        "take_profit": [r.setup.take_profit for r in results],
        "exit_price": [r.exit_price for r in results],
        "outcome": [r.outcome for r in results],
        "pnl": [r.pnl for r in results],
        "pnl_percent": [r.pnl_percent for r in results],
        "r_multiple": [r.r_multiple for r in results],
        "bars_held": [r.bars_held for r in results],
        "confidence": [r.setup.confidence for r in results],
    }

    results_df = pl.DataFrame(results_data)

    # Calculate summary statistics
    logger.info("=" * 80)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Trades: {len(results_df)}")
    logger.info(f"Wins: {len(results_df.filter(pl.col('outcome') == 'win'))}")
    logger.info(f"Losses: {len(results_df.filter(pl.col('outcome') == 'loss'))}")
    logger.info(
        f"Win Rate: {len(results_df.filter(pl.col('outcome') == 'win')) / len(results_df):.2%}"
    )
    logger.info(f"Average PnL: ${results_df['pnl'].mean():.2f}")
    logger.info(f"Average R-Multiple: {results_df['r_multiple'].mean():.2f}R")
    logger.info(f"Total PnL: ${results_df['pnl'].sum():.2f}")
    logger.info(f"Profit Factor: {calculate_profit_factor(results_df):.2f}")
    logger.info("=" * 80)

    return results_df


def backtest_single_trade(
    df: pl.DataFrame, setup: TradeSetup, timezone: str = "US/Eastern"
) -> SignalResult | None:
    """
    Backtest a single trade setup.

    Args:
        df: Full DataFrame with price data
        setup: TradeSetup to backtest
        timezone: Timezone for the data

    Returns:
        SignalResult or None if trade cannot be backtested
    """
    # Filter data for this date and after entry time (11:00)
    trade_data = df.filter(
        (pl.col("date") == setup.date) & (pl.col("DateTime").dt.hour() >= 11)
    )

    if len(trade_data) == 0:
        logger.warning(f"No post-entry data for date {setup.date}")
        return None

    # Simulate trade execution
    if setup.signal == SignalType.BUY:
        outcome, exit_price, bars_held = simulate_long_trade(
            trade_data, setup.entry_price, setup.stop_loss, setup.take_profit
        )
    elif setup.signal == SignalType.SELL:
        outcome, exit_price, bars_held = simulate_short_trade(
            trade_data, setup.entry_price, setup.stop_loss, setup.take_profit
        )
    else:
        logger.warning(f"Cannot backtest HOLD signal for date {setup.date}")
        return None

    # Calculate PnL
    if setup.signal == SignalType.BUY:
        pnl = exit_price - setup.entry_price
    else:  # SELL
        pnl = setup.entry_price - exit_price

    pnl_percent = pnl / setup.entry_price

    # Create result
    result = SignalResult(
        setup=setup,
        outcome=outcome,
        exit_price=exit_price,
        pnl=pnl,
        pnl_percent=pnl_percent,
        bars_held=bars_held,
    )

    return result


def simulate_long_trade(
    trade_data: pl.DataFrame, entry: float, stop_loss: float, take_profit: float
) -> tuple[Literal["win", "loss", "breakeven", "open"], float, int]:
    """
    Simulate a long trade (BUY signal).

    Args:
        trade_data: Price data after entry
        entry: Entry price
        stop_loss: Stop-loss price
        take_profit: Take-profit price

    Returns:
        Tuple of (outcome, exit_price, bars_held)
    """
    for i, row in enumerate(trade_data.iter_rows(named=True)):
        # Check if stop-loss hit
        if row["Low"] <= stop_loss:
            return "loss", stop_loss, i + 1

        # Check if take-profit hit
        if row["High"] >= take_profit:
            return "win", take_profit, i + 1

    # Trade still open at end of day - exit at close
    final_close = trade_data["Close"][-1]
    if final_close > entry:
        outcome = "win"
    elif final_close < entry:
        outcome = "loss"
    else:
        outcome = "breakeven"

    return outcome, final_close, len(trade_data)


def simulate_short_trade(
    trade_data: pl.DataFrame, entry: float, stop_loss: float, take_profit: float
) -> tuple[Literal["win", "loss", "breakeven", "open"], float, int]:
    """
    Simulate a short trade (SELL signal).

    Args:
        trade_data: Price data after entry
        entry: Entry price
        stop_loss: Stop-loss price
        take_profit: Take-profit price

    Returns:
        Tuple of (outcome, exit_price, bars_held)
    """
    for i, row in enumerate(trade_data.iter_rows(named=True)):
        # Check if stop-loss hit (price goes up)
        if row["High"] >= stop_loss:
            return "loss", stop_loss, i + 1

        # Check if take-profit hit (price goes down)
        if row["Low"] <= take_profit:
            return "win", take_profit, i + 1

    # Trade still open at end of day - exit at close
    final_close = trade_data["Close"][-1]
    if final_close < entry:
        outcome = "win"
    elif final_close > entry:
        outcome = "loss"
    else:
        outcome = "breakeven"

    return outcome, final_close, len(trade_data)


def calculate_profit_factor(results_df: pl.DataFrame) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        results_df: DataFrame with backtest results

    Returns:
        Profit factor
    """
    wins = results_df.filter(pl.col("outcome") == "win")
    losses = results_df.filter(pl.col("outcome") == "loss")

    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_sharpe_ratio(
    results_df: pl.DataFrame, risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio of returns.

    Args:
        results_df: DataFrame with backtest results
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    returns = results_df["pnl_percent"]
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    # Annualize (assuming ~252 trading days)
    sharpe = (mean_return - risk_free_rate) / std_return * (252**0.5)

    return sharpe


def calculate_max_drawdown(results_df: pl.DataFrame) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown from cumulative PnL.

    Args:
        results_df: DataFrame with backtest results

    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    cumulative_pnl = results_df["pnl"].cum_sum()
    running_max = cumulative_pnl.cum_max()
    drawdown = cumulative_pnl - running_max

    max_dd = drawdown.min()
    max_dd_idx = drawdown.arg_min()

    # Find start of drawdown (last peak before max drawdown)
    start_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if drawdown[i] == 0:
            start_idx = i
            break

    return max_dd, start_idx, max_dd_idx


def generate_performance_report(results_df: pl.DataFrame) -> dict:
    """
    Generate comprehensive performance report.

    Args:
        results_df: DataFrame with backtest results

    Returns:
        Dictionary with performance metrics
    """
    total_trades = len(results_df)
    wins = len(results_df.filter(pl.col("outcome") == "win"))
    losses = len(results_df.filter(pl.col("outcome") == "loss"))

    win_rate = wins / total_trades if total_trades > 0 else 0
    avg_win = (
        results_df.filter(pl.col("outcome") == "win")["pnl"].mean() if wins > 0 else 0
    )
    avg_loss = (
        results_df.filter(pl.col("outcome") == "loss")["pnl"].mean()
        if losses > 0
        else 0
    )

    profit_factor = calculate_profit_factor(results_df)
    sharpe_ratio = calculate_sharpe_ratio(results_df)
    max_dd, dd_start, dd_end = calculate_max_drawdown(results_df)

    report = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_pnl": results_df["pnl"].mean(),
        "total_pnl": results_df["pnl"].sum(),
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
        "avg_r_multiple": results_df["r_multiple"].mean(),
        "avg_bars_held": results_df["bars_held"].mean(),
    }

    return report


# Made with Bob
