#!/usr/bin/env python3
"""
Chimera Backtesting Examples
Quick reference for common backtesting commands
"""

def show_examples():
    """Display example commands"""
    
    print("Chimera Backtesting Examples")
    print("=" * 50)
    
    examples = [
        {
            'title': 'Basic Momentum Test (30 days)',
            'command': 'python run_backtest.py TSLA m 30',
            'description': 'Test TSLA with momentum strategy over last 30 days'
        },
        {
            'title': 'Value Strategy (6 months)',
            'command': 'python run_backtest.py AAPL v 180',
            'description': 'Test AAPL with value strategy over 6 months'
        },
        {
            'title': 'Custom Date Range',
            'command': 'python run_backtest.py NVDA m 2023-01-20 2023-07-20',
            'description': 'Test NVDA momentum strategy for specific period'
        },
        {
            'title': 'High Capital Test',
            'command': 'python run_backtest.py MSFT m 90 50000',
            'description': 'Test MSFT with $50,000 starting capital'
        },
        {
            'title': 'Full Analysis (Plot + Save)',
            'command': 'python run_backtest.py TSLA m 60 --plot --save',
            'description': 'Run backtest with charts and save results to file'
        },
        {
            'title': 'Quick Comparison',
            'command': 'python run_backtest.py AAPL m 30 && python run_backtest.py AAPL v 30',
            'description': 'Compare momentum vs value strategies'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
    
    print(f"\n{'='*50}")
    print("Parameter Reference:")
    print("  <ticker>    : Stock symbol (TSLA, AAPL, NVDA, etc.)")
    print("  <strategy>  : m/momentum or v/value")
    print("  <period>    : Days (30, 90, 180) OR date range (2023-01-20 2023-07-20)")
    print("  [cash]      : Starting capital (default: $10,000)")
    print("  --plot      : Show performance charts")
    print("  --save      : Save results to JSON file")
    
    print(f"\nPro Tips:")
    print("  • Use historical dates (15+ days old) to avoid API limits")
    print("  • Momentum works best with 30-90 day periods")
    print("  • Value works best with 90-180 day periods")
    print("  • Test multiple tickers to validate strategy robustness")
    
    print(f"\n🔄 Batch Testing:")
    print("  python batch_backtest.py    # Run multiple configurations")

if __name__ == "__main__":
    show_examples()