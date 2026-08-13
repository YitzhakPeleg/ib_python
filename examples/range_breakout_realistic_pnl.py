"""Realistic $-P&L simulation of the range-breakout strategy's best config.

Every other result in this project reports 1-share/contract P&L, gross of
costs — this script instead runs an actual account simulation: starting
capital, per-trade position sizing (risk a fixed % of current equity,
sized off the SL distance -- shares floored, capped so position value
never exceeds available equity, i.e. no margin), IBKR-style commissions
(COMMISSION_MIN per leg, or COMMISSION_PER_SHARE * shares if that's
higher -- whichever is bigger, charged on BOTH the entry and exit leg of
each round-trip trade), and a flat per-share slippage assumption applied
unfavorably on both legs.

Trades are the config from docs/04_range_breakout_strategy.md Appendix G
(the current best: 1-minute trigger scan, range < ATR/2, TP=ATR/7,
SL=ATR/5, min_relative_volume=0.08, no require_open_outside), run
sequentially in trigger-time order so the equity curve — and therefore
position sizing — compounds realistically trade by trade (never more than
one position open at a time, matching the strategy's own rule).
"""

import polars as pl
from loguru import logger

from algo.range_breakout import run_breakout_backtest
from algo.resample_bars import resample_to_timeframe
from models.paths import get_file

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
TICKER = "SPY"
DATA_LABEL = "SPY_full"

STARTING_CAPITAL = 10_000.0
COMMISSION_MIN = 2.50  # $ per leg
COMMISSION_PER_SHARE = 0.01  # $ per share per leg, if that exceeds COMMISSION_MIN
SLIPPAGE_PER_SHARE = 0.01  # $ per share, unfavorable, applied on both legs
RISK_PCT_PER_TRADE = 0.01  # fraction of current equity risked per trade


def load_trades() -> pl.DataFrame:
    minute_bars_full = (
        pl.read_parquet(get_file(DATA_LABEL, "1_min"))
        .select(OHLCV)
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .sort("DateTime")
    )
    daily_bars = (
        resample_to_timeframe(minute_bars_full.drop("ticker"), "1d")
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .select(OHLCV + ["ticker"])
    )
    trades = run_breakout_backtest(
        minute_bars_full,
        daily_bars,
        signal_timeframe="1m",
        opening_range_minutes=30,
        signal_window_minutes=210,
        range_atr_low_divisor=None,
        range_atr_high_divisor=2.0,
        tp_atr_divisor=7.0,
        sl_atr_divisor=5.0,
        min_relative_volume=0.08,
        allow_multiple_trades_per_day=True,
    )
    return trades.filter(pl.col("date").dt.year() <= 2025).sort("trigger_time")


def simulate(trades: pl.DataFrame) -> pl.DataFrame:
    equity = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    rows = []
    skipped_zero_shares = 0

    for row in trades.iter_rows(named=True):
        entry = row["entry_price"]
        sl = row["sl"]
        exit_price = row["exit_price"]
        direction = row["direction"]

        risk_per_share = abs(entry - sl)
        if risk_per_share <= 0:
            continue

        risk_dollars = equity * RISK_PCT_PER_TRADE
        shares = int(risk_dollars / risk_per_share)
        max_shares_by_capital = int(equity / entry)  # no margin
        shares = min(shares, max_shares_by_capital)
        if shares < 1:
            skipped_zero_shares += 1
            continue

        entry_eff = entry + SLIPPAGE_PER_SHARE * direction
        exit_eff = exit_price - SLIPPAGE_PER_SHARE * direction
        gross_pnl = (exit_eff - entry_eff) * direction * shares

        commission = 2 * max(COMMISSION_MIN, COMMISSION_PER_SHARE * shares)
        net_pnl = gross_pnl - commission

        equity += net_pnl
        peak = max(peak, equity)
        drawdown = peak - equity
        drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0.0

        rows.append(
            {
                "date": row["date"],
                "year": row["date"].year,
                "direction": direction,
                "shares": shares,
                "gross_pnl": gross_pnl,
                "commission": commission,
                "net_pnl": net_pnl,
                "equity": equity,
                "drawdown": drawdown,
                "drawdown_pct": drawdown_pct,
            }
        )

    if skipped_zero_shares:
        logger.info(
            f"{skipped_zero_shares} trades skipped (position size rounded to 0 shares)"
        )
    return pl.DataFrame(rows)


def report(sim: pl.DataFrame) -> None:
    print(f"Starting capital: ${STARTING_CAPITAL:,.2f}")
    print(
        f"Assumptions: risk {RISK_PCT_PER_TRADE:.0%} of equity/trade (sized off SL "
        f"distance, capped at no-margin buying power), commission = "
        f"max(${COMMISSION_MIN:.2f}, ${COMMISSION_PER_SHARE:.2f}/share) per leg, "
        f"slippage = ${SLIPPAGE_PER_SHARE:.2f}/share per leg\n"
    )

    by_year = (
        sim.group_by("year")
        .agg(
            pl.len().alias("n"),
            pl.col("gross_pnl").sum().alias("gross_pnl"),
            pl.col("commission").sum().alias("commission"),
            pl.col("net_pnl").sum().alias("net_pnl"),
            pl.col("drawdown_pct").max().alias("max_dd_pct_in_year"),
        )
        .sort("year")
    )
    print("Per-year:")
    print(by_year)
    print()

    n_total = sim.height
    gross_total = float(sim["gross_pnl"].sum())
    commission_total = float(sim["commission"].sum())
    net_total = float(sim["net_pnl"].sum())
    final_equity = STARTING_CAPITAL + net_total
    total_return_pct = net_total / STARTING_CAPITAL * 100
    max_dd = float(sim["drawdown"].max())
    max_dd_pct = float(sim["drawdown_pct"].max())
    wins = int((sim["net_pnl"] > 0).sum())
    win_pct = wins / n_total * 100

    print("=== Overall (2018-2025) ===")
    print(f"Trades executed:      {n_total}")
    print(f"Win rate (net of costs): {win_pct:.1f}%")
    print(f"Gross P&L:             ${gross_total:,.2f}")
    print(f"Total commission:      ${commission_total:,.2f}")
    print(
        f"Total slippage cost:   ${gross_total - (net_total + commission_total):,.2f}"
    )
    print(f"Net P&L:               ${net_total:,.2f}")
    print(f"Final equity:          ${final_equity:,.2f}")
    print(f"Total return:          {total_return_pct:+.1f}%")
    print(f"Max drawdown:          ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
    print(f"Avg shares/trade:      {sim['shares'].mean():.1f}")
    print(f"Avg commission/trade:  ${commission_total / n_total:.2f}")


if __name__ == "__main__":
    trades = load_trades()
    sim = simulate(trades)
    report(sim)
