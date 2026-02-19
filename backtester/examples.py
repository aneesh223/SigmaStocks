#!/usr/bin/env python3
"""
Orthrus Backtesting Examples
Quick reference for common backtesting commands with optimal timeframes
"""

def show_examples():
    """Display example commands"""
    
    print("Orthrus Backtesting Examples")
    print("=" * 80)
    
    print("\n📊 OPTIMAL BACKTESTING TIMEFRAMES")
    print("-" * 80)
    print("MOMENTUM Strategy: 3-5 years (default: 5 years)")
    print("  Why: • Requires ~10 months to warm up 200-day SMA for Golden/Death Cross")
    print("       • Captures bull runs, bear markets, and choppy sideways periods")
    print("       • Provides 100+ trades for statistical significance")
    print("\nVALUE Strategy: 5-10+ years (default: 10 years)")
    print("  Why: • Tests long-term mean reversion across full market cycles")
    print("       • Validates conservative risk management during actual crashes")
    print("       • Captures multiple bull/bear regime transitions")
    print("=" * 80)
    
    examples = [
        {
            'title': 'Momentum Strategy (Default 5 Years)',
            'command': 'python run_backtest.py TSLA m',
            'description': 'Test TSLA with momentum strategy using optimal 5-year period'
        },
        {
            'title': 'Value Strategy (Default 10 Years)',
            'command': 'python run_backtest.py AAPL v',
            'description': 'Test AAPL with value strategy using optimal 10-year period'
        },
        {
            'title': 'Momentum - Minimum Recommended (3 Years)',
            'command': 'python run_backtest.py NVDA m 1095',
            'description': 'Test NVDA momentum with 3-year period (1095 days)'
        },
        {
            'title': 'Value - Extended Period (15 Years)',
            'command': 'python run_backtest.py MSFT v 5475',
            'description': 'Test MSFT value with 15-year period for maximum cycle coverage'
        },
        {
            'title': 'Custom Date Range',
            'command': 'python run_backtest.py GOOGL m 2019-01-01 2024-01-01',
            'description': 'Test GOOGL momentum for specific 5-year period'
        },
        {
            'title': 'High Capital Test',
            'command': 'python run_backtest.py AMZN v 3650 50000',
            'description': 'Test AMZN value with $50,000 starting capital over 10 years'
        },
        {
            'title': 'Full Analysis (Plot + Save)',
            'command': 'python run_backtest.py TSLA m 1825 --plot --save',
            'description': 'Run 5-year momentum backtest with charts and save results'
        },
        {
            'title': 'Strategy Comparison',
            'command': 'python run_backtest.py AAPL m 1825 && python run_backtest.py AAPL v 3650',
            'description': 'Compare momentum (5yr) vs value (10yr) with optimal periods'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
    
    print(f"\n{'='*80}")
    print("Parameter Reference:")
    print("  <ticker>    : Stock symbol (TSLA, AAPL, NVDA, etc.)")
    print("  <strategy>  : m/momentum or v/value")
    print("  [period]    : OPTIONAL - Days (1095, 1825, 3650) OR date range")
    print("                If omitted, uses strategy default (momentum=5yr, value=10yr)")
    print("  [cash]      : Starting capital (default: $10,000)")
    print("  --plot      : Show performance charts")
    print("  --no-plot   : Disable charts (useful for batch testing)")
    print("  --save      : Save results to JSON file")
    
    print(f"\n💡 Pro Tips:")
    print("  • Always use strategy defaults unless testing specific hypotheses")
    print("  • Momentum: 1095 days (3yr) minimum, 1825 days (5yr) recommended")
    print("  • Value: 1825 days (5yr) minimum, 3650 days (10yr) recommended")
    print("  • Use historical dates (15+ days old) to avoid API rate limits")
    print("  • Test multiple tickers to validate strategy robustness")
    print("  • Longer periods = more statistical significance")
    
    print(f"\n🔄 Batch Testing:")
    print("  python batch_backtest.py           # Run 10 random backtests (default)")
    print("  python batch_backtest.py 25        # Run 25 random backtests")
    print("  python batch_backtest.py 50 --seed 123  # Run 50 tests with reproducible seed")
    print("\n  Note: Batch tests use randomized periods (30-365 days) for diversity testing")
    print("        For production validation, use individual tests with optimal periods")

if __name__ == "__main__":
    show_examples()