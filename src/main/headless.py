#!/usr/bin/env python3
"""
Orthrus Headless Analysis CLI
Usage: python headless.py <ticker> <strategy> <period> [--save]
"""

import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import logging

# Force no-UI backend
matplotlib.use('Agg')

# Monkey patch: Disable plt.show() completely
plt.show = lambda *args, **kwargs: None

from datetime import datetime
import pytz

# Add parent directories to python path to import internal modules
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
root_dir = os.path.join(parent_dir, '..')
sys.path.insert(0, parent_dir)  # Add src/ to path
sys.path.insert(0, root_dir)    # Add project root to path

try:
    from input import analyze_stock
except ImportError:
    logging.error("Error: Could not import 'input.py'. Make sure you're running from the correct directory.")
    logging.error("   Try: python src/main/headless.py <ticker> <strategy> <period>")
    sys.exit(1)

def parse_period(period_arg, next_arg=None):
    try:
        days = int(period_arg)
        return days, 1 
    except ValueError:
        if next_arg is None:
            raise ValueError("When using date format, both start and end dates are required")
        try:
            start_date = pytz.utc.localize(datetime.strptime(period_arg, "%Y-%m-%d"))
            days_delta = (datetime.now(pytz.utc) - start_date).days
            return max(1, days_delta), 2
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

def main():
    if len(sys.argv) < 3:
        print("Usage: python headless.py <ticker> <strategy> <period> [--save]")
        print("Example: python headless.py TSLA m 30 --save")
        return 1
    
    ticker = sys.argv[1].upper()
    
    strategy_arg = sys.argv[2].lower()
    strategy = 'momentum' if strategy_arg in ['m', 'momentum'] else 'value'
    
    # Parse period
    if len(sys.argv) > 3 and not sys.argv[3].startswith('--'):
        try:
            lookback_days, args_consumed = parse_period(sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        except (ValueError, IndexError):
            lookback_days = 30
    else:
        lookback_days = 30
    
    # Parse flags
    # We check all arguments for the flag
    save_plot = '--save' in sys.argv
    
    print(f"\nOrthrus Headless Analysis: {ticker}")
    print("-" * 50)
    
    try:
        # Run Analysis
        analyze_stock(ticker, lookback_days, strategy)
        
        if save_plot:
            # Manually save the current figure since show() didn't clear it
            filename = f"analysis_{ticker}_{strategy}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename)
            logging.info(f"\nPlot saved to: {filename}")
            
        logging.info(f"\nAnalysis Complete.")
        return 0
        
    except KeyboardInterrupt:
        logging.info("\nAnalysis interrupted.")
        return 0
    except Exception as e:
        logging.error(f"\nFatal Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())