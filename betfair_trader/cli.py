"""CLI entry point for the Betfair trading bot."""

from __future__ import annotations

import argparse
import logging
import sys

from betfair_trader.config import BetfairConfig
from betfair_trader.engine import TradingEngine
from betfair_trader.storage import TradeLogger


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Start the trading engine."""
    config = BetfairConfig()
    if args.paper:
        config.paper_trading = True
    if args.live:
        config.paper_trading = False

    engine = TradingEngine(config)
    engine.start()


def cmd_status(args: argparse.Namespace) -> None:
    """Show trading performance summary."""
    logger = TradeLogger()
    summary = logger.get_performance_summary()
    logger.close()

    if not summary or summary.get("total_trades", 0) == 0:
        print("No trades recorded yet.")
        return

    print("\n=== Trading Performance ===")
    print(f"Total Trades:   {summary['total_trades']}")
    print(f"Winning:        {summary['winning_trades']}")
    print(f"Losing:         {summary['losing_trades']}")
    print(f"Total P&L:      £{summary['total_pnl']:.2f}")
    print(f"Avg P&L:        £{summary['avg_pnl']:.2f}")
    print(f"Avg Edge:       {summary['avg_edge']:.3f}")
    print(f"Avg Predicted P: {summary['avg_predicted_p']:.3f}")


def cmd_bias(args: argparse.Namespace) -> None:
    """Show bias analysis from the feedback loop."""
    logger = TradeLogger()
    report = logger.get_bias_report()
    logger.close()

    if not report:
        print("No completed trades with outcomes to analyze.")
        return

    print("\n=== Bias Report ===")
    for row in report:
        print(f"\nStrategy: {row['strategy']} | Signal: {row['signal']}")
        print(f"  Trades:          {row['count']}")
        print(f"  Predicted P:     {row['avg_predicted_p']:.3f}")
        print(f"  Actual Win Rate: {row['actual_win_rate']:.3f}")
        diff = row['actual_win_rate'] - row['avg_predicted_p']
        bias = "OVERCONFIDENT" if diff < -0.05 else "UNDERCONFIDENT" if diff > 0.05 else "CALIBRATED"
        print(f"  Calibration:     {bias} ({diff:+.3f})")
        print(f"  Total P&L:       £{row['total_pnl']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="betfair-trader",
        description="AI-powered Betfair Exchange trading bot",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Start the trading engine")
    run_parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    run_parser.add_argument("--live", action="store_true", help="Force live trading mode")

    # Status command
    subparsers.add_parser("status", help="Show trading performance")

    # Bias command
    subparsers.add_parser("bias", help="Show prediction bias analysis")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "bias":
        cmd_bias(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
