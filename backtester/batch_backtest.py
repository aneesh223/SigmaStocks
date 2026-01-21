#!/usr/bin/env python3
"""
SigmaStocks Batch Backtesting
Run multiple backtests quickly for comparison

Usage: python batch_backtest.py
"""

import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

def run_backtest(ticker, strategy, date_range, cash=10000):
    """Run a single backtest using the CLI (explicitly disable plotting for batch mode)"""
    # Change to the backtester directory to run the command
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse date range
    if ' ' in date_range:
        start_date, end_date = date_range.split(' ')
        cmd = ['python3', 'backtest.py', ticker, strategy, start_date, end_date, str(cash), '--no-plot']
    else:
        cmd = ['python3', 'backtest.py', ticker, strategy, str(date_range), str(cash), '--no-plot']
    # Explicitly add --no-plot flag for batch mode
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    try:
        # Run from the backtester directory
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=script_dir)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return False
    finally:
        # Restore original directory
        os.chdir(original_dir)

def main():
    """Run batch backtests"""
    
    print("🚀 SigmaStocks Batch Backtesting")
    print("Testing multiple configurations...")
    
    # Define test configurations (using historical dates that work with free Alpaca)
    configs = [
        # Format: (ticker, strategy, start_date, end_date, cash)
        ('TSLA', 'm', '2023-01-20', '2023-07-20', 10000),
        ('TSLA', 'v', '2023-01-20', '2023-07-20', 10000),
        ('AAPL', 'm', '2023-01-20', '2023-07-20', 10000),
        ('AAPL', 'v', '2023-01-20', '2023-07-20', 10000),
        ('NVDA', 'm', '2023-01-20', '2023-07-20', 10000),
        ('MSFT', 'v', '2023-01-20', '2023-07-20', 10000),
    ]
    
    successful = 0
    total = len(configs)
    
    for ticker, strategy, start_date, end_date, cash in configs:
        if run_backtest(ticker, strategy, f"{start_date} {end_date}", cash):
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"📊 BATCH RESULTS: {successful}/{total} successful")
    print('='*60)
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    exit(main())