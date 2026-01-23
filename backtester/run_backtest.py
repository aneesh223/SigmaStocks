#!/usr/bin/env python3
"""
Orthrus Quick Backtesting CLI
Usage: python run_backtest.py <ticker> <strategy> <period> [cash] [--plot] [--save]

Examples:
  python run_backtest.py TSLA m 5           # TSLA, momentum, last 5 periods (days)
  python run_backtest.py AAPL v 90          # AAPL, value, last 90 days  
  python run_backtest.py NVDA m 2023-01-20 2023-07-20  # Custom date range
  python run_backtest.py TSLA m 30 15000 --plot --save  # With custom cash, plot, and save
"""

import sys
import os
import pytz
import argparse
from datetime import datetime, timedelta

# Add parent directory to path to import src modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import the backtester class directly from the module file
sys.path.insert(0, os.path.dirname(__file__))
from backtester import AlpacaBacktester

def parse_period(period_arg, next_arg=None):
    """
    Parse period argument which can be:
    - Number (days): 30, 90, 180
    - Date range: 2023-01-20 (requires next_arg as end date)
    """
    try:
        # Try to parse as number of days
        days = int(period_arg)
        # Use 1 year ago as end date for more stable historical data
        end_date = datetime.now(pytz.utc) - timedelta(days=365)
        start_date = end_date - timedelta(days=days)
        return start_date, end_date, 1  # Return 1 to indicate we used 1 argument
    except ValueError:
        # Try to parse as start date (expecting end date as next argument)
        if next_arg is None:
            raise ValueError("When using date format, both start and end dates are required")
        
        try:
            start_date = pytz.utc.localize(datetime.strptime(period_arg, "%Y-%m-%d"))
            end_date = pytz.utc.localize(datetime.strptime(next_arg, "%Y-%m-%d"))
            return start_date, end_date, 2  # Return 2 to indicate we used 2 arguments
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

def main():
    """Command-line backtesting interface"""
    
    if len(sys.argv) < 4:
        print("❌ Usage: python run_backtest.py <ticker> <strategy> <period> [cash] [--plot] [--no-plot] [--save]")
        print("\nExamples:")
        print("  python run_backtest.py TSLA m 30")
        print("  python run_backtest.py AAPL v 90")
        print("  python run_backtest.py NVDA m 2023-01-20 2023-07-20")
        print("  python run_backtest.py TSLA m 30 15000 --no-plot --save")
        print("\nStrategy: m/momentum or v/value")
        print("Period: number of days OR start-date end-date")
        print("Flags: --plot (force plot), --no-plot (disable plot), --save (save results)")
        print("Note: Plots are shown by default for single backtests")
        return 1
    
    # Parse arguments
    ticker = sys.argv[1].upper()
    
    # Parse strategy
    strategy_arg = sys.argv[2].lower()
    if strategy_arg in ['m', 'momentum']:
        strategy = 'momentum'
    elif strategy_arg in ['v', 'value']:
        strategy = 'value'
    else:
        print("❌ Strategy must be 'm'/'momentum' or 'v'/'value'")
        return 1
    
    # Parse period (might consume 1 or 2 arguments)
    try:
        start_date, end_date, args_consumed = parse_period(sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    except (ValueError, IndexError) as e:
        print(f"❌ Period error: {e}")
        return 1
    
    # Parse remaining arguments
    remaining_args = sys.argv[3 + args_consumed:]
    
    # Parse cash (optional)
    initial_cash = 10000.0
    cash_idx = 0
    if remaining_args and not remaining_args[0].startswith('--'):
        try:
            initial_cash = float(remaining_args[0])
            cash_idx = 1
        except ValueError:
            pass
    
    # Parse flags
    show_plot = '--plot' in remaining_args[cash_idx:] or '--no-plot' not in remaining_args[cash_idx:]  # Default to True unless --no-plot
    save_results = '--save' in remaining_args[cash_idx:]
    
    # Display configuration
    print(f"🚀 Orthrus Backtest")
    print(f"📊 {ticker} | {strategy.upper()} | {start_date.date()} to {end_date.date()} | ${initial_cash:,.0f}")
    
    try:
        # Run backtest
        backtester = AlpacaBacktester(ticker, strategy, initial_cash)
        results = backtester.run(start_date, end_date, lookback_days=30)
        
        # Show plot by default (unless --no-plot is specified)
        if show_plot:
            backtester.plot_results(results)
        
        # Save results if requested
        if save_results:
            import json
            filename = f"backtest_{ticker}_{strategy}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            
            # Convert datetime objects to strings for JSON serialization
            results_copy = results.copy()
            results_copy['start_date'] = results_copy['start_date'].isoformat()
            results_copy['end_date'] = results_copy['end_date'].isoformat()
            
            # Convert portfolio history dates
            for item in results_copy['portfolio_history']:
                item['Date'] = item['Date'].isoformat()
            
            # Convert trade log dates
            for item in results_copy['trade_log']:
                item['Date'] = item['Date'].isoformat()
            
            with open(filename, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {filename}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())