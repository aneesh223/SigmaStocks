#!/usr/bin/env python3
"""
Orthrus Random Batch Backtesting
Randomly selects from pools of tickers, strategies, and date ranges for comprehensive testing
Features parallel execution with configurable threading for faster batch processing
"""

import subprocess
import sys
import os
import random
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def get_random_date_range(strategy=None):
    """
    Generate random start and end dates within Alpaca free tier support.
    
    If strategy is provided, uses strategy-optimal periods:
    - MOMENTUM: 3-5 years (1095-1825 days)
    - VALUE: 5-10 years (1825-3650 days)
    
    Otherwise uses diverse testing periods (30-365 days) for comprehensive validation.
    
    Alpaca free tier supports data from ~2015 onwards.
    We'll use 2015-2024 for good data quality and recent market conditions.
    """
    # Define the available date range (Alpaca free tier with good data quality)
    start_year = 2015
    end_year = 2024
    
    # Strategy-specific optimal periods
    if strategy:
        if strategy.lower() in ['m', 'momentum']:
            # Momentum: 3-5 years
            min_days = 1095  # 3 years
            max_days = 1825  # 5 years
        else:  # value
            # Value: 5-10 years
            min_days = 1825  # 5 years
            max_days = 3650  # 10 years
    else:
        # Diverse testing: 30-365 days for comprehensive validation
        min_days = 30
        max_days = 365
    
    # Generate random start date
    start_date = datetime(
        year=random.randint(start_year, end_year - 1),
        month=random.randint(1, 12),
        day=random.randint(1, 28)  # Use 28 to avoid month-end issues
    )
    
    # Generate random period length
    period_days = random.randint(min_days, max_days)
    
    end_date = start_date + timedelta(days=period_days)
    
    # Ensure end date doesn't exceed our range
    if end_date.year > end_year:
        end_date = datetime(end_year, 12, 31)
    
    # Ensure we don't go into the future (leave some buffer)
    max_end_date = datetime.now() - timedelta(days=15)  # 15 days buffer for API limits
    if end_date > max_end_date:
        end_date = max_end_date
        # Adjust start date if needed
        if (end_date - start_date).days < min_days:
            start_date = end_date - timedelta(days=min_days)
    
    # Ensure start date is not before our data range
    min_start_date = datetime(start_year, 1, 1)
    if start_date < min_start_date:
        start_date = min_start_date
        end_date = start_date + timedelta(days=period_days)
        if end_date > max_end_date:
            end_date = max_end_date
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_ticker_pools():
    """Define pools of tickers by category for diverse testing"""
    return {
        'mega_cap_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'large_cap_tech': ['NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'AVGO'],
        'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B'],
        'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR'],
        'consumer': ['KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX'],
        'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO'],
        'industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX'],
        'volatile_growth': ['TSLA', 'NVDA', 'AMD', 'NFLX', 'ZOOM', 'ROKU', 'PLTR']
    }

def select_random_parameters(use_optimal_periods=False):
    """
    Randomly select ticker, strategy, and date range.
    
    Args:
        use_optimal_periods: If True, uses strategy-optimal periods (3-10 years)
                           If False, uses diverse testing periods (30-365 days)
    """
    ticker_pools = get_ticker_pools()
    
    # Randomly select a category and then a ticker from that category
    category = random.choice(list(ticker_pools.keys()))
    ticker = random.choice(ticker_pools[category])
    
    # Randomly select strategy
    strategy = random.choice(['m', 'v'])  # momentum or value
    
    # Generate date range (strategy-aware if optimal periods enabled)
    if use_optimal_periods:
        start_date, end_date = get_random_date_range(strategy)
    else:
        start_date, end_date = get_random_date_range()
    
    # Create description
    strategy_name = "MOMENTUM" if strategy == 'm' else "VALUE"
    period_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    period_years = period_days / 365.25
    description = f"RANDOM TEST: {ticker} {strategy_name} strategy ({category.replace('_', ' ').title()}) - {period_years:.1f} years"
    
    return ticker, strategy, start_date, end_date, description, category

def run_backtest(ticker, strategy, start_date, end_date, description="", print_header=True):
    """Run a single backtest and capture results"""
    cmd = ['python3', 'run_backtest.py', ticker, strategy, start_date, end_date, '--no-plot']
    
    if print_header:
        print(f"\n{'='*80}")
        print(f"TEST: {ticker} {strategy.upper()} | {start_date} to {end_date}")
        if description:
            print(f"{description}")
        print('='*80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            strategy_return = None
            buy_hold_return = None
            alpha = None
            dollar_alpha = None
            relative_alpha = None
            strategy_gain = None
            buy_hold_gain = None
            trades = None
            regime = None
            
            for line in lines:
                if "Strategy Return:" in line:
                    strategy_return = line.split(":")[1].strip()
                elif "Buy & Hold Return:" in line:
                    buy_hold_return = line.split(":")[1].strip()
                elif "Alpha (Percentage Points):" in line:
                    alpha = line.split(":")[1].strip()
                elif "Dollar Alpha:" in line:
                    dollar_alpha = line.split(":")[1].strip()
                elif "Relative Alpha:" in line:
                    relative_alpha = line.split(":")[1].strip().split()[0]  # Get just the percentage
                elif "Strategy Gain:" in line:
                    strategy_gain = line.split(":")[1].strip()
                elif "Buy-Hold Gain:" in line:
                    buy_hold_gain = line.split(":")[1].strip()
                elif "Number of Trades:" in line:
                    trades = line.split(":")[1].strip()
                elif "Initial Regime:" in line:
                    regime = line.split(":")[1].strip()
            
            return {
                'success': True,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'alpha': alpha,
                'dollar_alpha': dollar_alpha,
                'relative_alpha': relative_alpha,
                'strategy_gain': strategy_gain,
                'buy_hold_gain': buy_hold_gain,
                'trades': trades,
                'regime': regime,
                'ticker': ticker,
                'strategy': strategy.upper(),
                'period': f"{start_date} to {end_date}",
                'description': description
            }
        else:
            print(f"FAILED: {result.stderr}")
            return {'success': False, 'ticker': ticker, 'strategy': strategy.upper(), 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Test took longer than 5 minutes")
        return {'success': False, 'ticker': ticker, 'strategy': strategy.upper(), 'error': 'Timeout'}
    except Exception as e:
        print(f"ERROR: {e}")
        return {'success': False, 'ticker': ticker, 'strategy': strategy.upper(), 'error': str(e)}

def main():
    """Run random batch tests with threading support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Orthrus Random Batch Backtesting')
    parser.add_argument('num_tests', type=int, nargs='?', default=10, 
                       help='Number of random backtests to run (default: 10)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--threads', type=int, default=4, 
                       help='Number of concurrent threads (default: 4)')
    parser.add_argument('--optimal-periods', action='store_true',
                       help='Use strategy-optimal periods (momentum=3-5yr, value=5-10yr) instead of diverse testing (30-365 days)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print("ORTHRUS RANDOM BATCH BACKTESTING")
    print(f"Running {args.num_tests} random backtests across diverse market conditions")
    if args.optimal_periods:
        print("Using OPTIMAL PERIODS: Momentum=3-5 years, Value=5-10 years")
    else:
        print("Using DIVERSE TESTING: 30-365 day periods for comprehensive validation")
    print(f"Using {args.threads} concurrent threads for parallel execution")
    print("=" * 80)
    
    # Generate all test parameters upfront for thread safety
    test_params = []
    for i in range(args.num_tests):
        ticker, strategy, start_date, end_date, description, category = select_random_parameters(args.optimal_periods)
        test_params.append((i+1, ticker, strategy, start_date, end_date, description, category))
    
    results = []
    successful = 0
    completed = 0
    
    # Thread-safe print lock
    print_lock = threading.Lock()
    
    def run_test_with_progress(params):
        """Wrapper function to run test with progress tracking"""
        nonlocal completed, successful
        
        test_num, ticker, strategy, start_date, end_date, description, category = params
        
        with print_lock:
            print(f"\n[TEST {test_num}/{args.num_tests}] Starting: {ticker} {strategy.upper()}")
            print(f"{'='*80}")
            print(f"TEST: {ticker} {strategy.upper()} | {start_date} to {end_date}")
            if description:
                print(f"{description}")
            print('='*80)
        
        # Run the backtest without printing header (we already printed it above)
        result = run_backtest(ticker, strategy, start_date, end_date, description, print_header=False)
        result['category'] = category  # Add category for analysis
        result['test_num'] = test_num
        
        with print_lock:
            completed += 1
            if result['success']:
                successful += 1
                # Show quick result
                alpha = result.get('alpha', 'N/A')
                trades = result.get('trades', 'N/A')
                print(f"✅ [{test_num}/{args.num_tests}] {ticker}: {alpha} alpha, {trades} trades ({completed}/{args.num_tests} complete)")
            else:
                print(f"❌ [{test_num}/{args.num_tests}] {ticker}: Failed - {result.get('error', 'Unknown error')} ({completed}/{args.num_tests} complete)")
        
        return result
    
    # Execute tests in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(run_test_with_progress, params): params for params in test_params}
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                params = future_to_params[future]
                test_num, ticker, strategy = params[0], params[1], params[2]
                with print_lock:
                    print(f"❌ [{test_num}/{args.num_tests}] {ticker}: Exception - {str(e)}")
                results.append({
                    'success': False, 
                    'ticker': ticker, 
                    'strategy': strategy.upper(), 
                    'error': str(e),
                    'test_num': test_num
                })
    
    # Sort results by test number for consistent output
    results.sort(key=lambda x: x.get('test_num', 0))
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"RANDOM BATCH TEST RESULTS")
    print(f"{'='*80}")
    print(f"Successful Tests: {successful}/{args.num_tests} ({successful/args.num_tests*100:.1f}%)")
    print()
    
    # Analyze results by category
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Group by category
        categories = {}
        for result in successful_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        # Print category analysis
        for category, cat_results in categories.items():
            print(f"\n{category.replace('_', ' ').upper()} ({len(cat_results)} tests)")
            print("-" * 60)
            
            for result in cat_results:
                alpha_color = "🟢" if result['alpha'] and float(result['alpha'].replace('%', '').replace('+', '')) > 0 else "🔴" if result['alpha'] and float(result['alpha'].replace('%', '').replace('+', '')) < -5 else "🟡"
                dollar_alpha_str = result.get('dollar_alpha', 'N/A')
                print(f"{alpha_color} {result['ticker']} {result['strategy']}: {result['alpha']} alpha ({dollar_alpha_str} $) | {result['trades']} trades | {result['regime']}")
                print(f"   Strategy: {result['strategy_return']} ({result.get('strategy_gain', 'N/A')}) vs Buy-Hold: {result['buy_hold_return']} ({result.get('buy_hold_gain', 'N/A')})")
                print(f"   Period: {result['period']}")
                print()
        
        # Calculate overall statistics
        alphas = []
        dollar_alphas = []
        relative_alphas = []
        momentum_alphas = []
        value_alphas = []
        momentum_dollar_alphas = []
        value_dollar_alphas = []
        
        for result in successful_results:
            if result['alpha']:
                try:
                    alpha_val = float(result['alpha'].replace('%', '').replace('+', ''))
                    alphas.append(alpha_val)
                    
                    if result['strategy'] == 'M':
                        momentum_alphas.append(alpha_val)
                    else:
                        value_alphas.append(alpha_val)
                except:
                    pass
            
            # Parse dollar alpha
            if result.get('dollar_alpha'):
                try:
                    dollar_val = float(result['dollar_alpha'].replace('$', '').replace(',', '').replace('+', ''))
                    dollar_alphas.append(dollar_val)
                    
                    if result['strategy'] == 'M':
                        momentum_dollar_alphas.append(dollar_val)
                    else:
                        value_dollar_alphas.append(dollar_val)
                except:
                    pass
            
            # Parse relative alpha
            if result.get('relative_alpha'):
                try:
                    rel_val = float(result['relative_alpha'].replace('%', '').replace('+', ''))
                    relative_alphas.append(rel_val)
                except:
                    pass
        
        if alphas:
            avg_alpha = sum(alphas) / len(alphas)
            positive_alpha_count = sum(1 for a in alphas if a > 0)
            near_zero_count = sum(1 for a in alphas if -2 <= a <= 2)
            
            print(f"\nOVERALL STATISTICS")
            print("-" * 40)
            print(f"Average Alpha (Percentage Points): {avg_alpha:+.2f}%")
            
            if dollar_alphas:
                avg_dollar_alpha = sum(dollar_alphas) / len(dollar_alphas)
                print(f"Average Dollar Alpha: ${avg_dollar_alpha:+,.2f}")
            
            if relative_alphas:
                avg_relative_alpha = sum(relative_alphas) / len(relative_alphas)
                print(f"Average Relative Alpha: {avg_relative_alpha:+.2f}% of initial capital")
            
            print(f"Positive Alpha: {positive_alpha_count}/{len(alphas)} ({positive_alpha_count/len(alphas)*100:.1f}%)")
            print(f"Near Buy-Hold (±2%): {near_zero_count}/{len(alphas)} ({near_zero_count/len(alphas)*100:.1f}%)")
            
            if momentum_alphas:
                momentum_avg = sum(momentum_alphas) / len(momentum_alphas)
                momentum_dollar_avg = sum(momentum_dollar_alphas) / len(momentum_dollar_alphas) if momentum_dollar_alphas else 0
                print(f"Momentum Strategy Avg: {momentum_avg:+.2f}% (${momentum_dollar_avg:+,.2f}) ({len(momentum_alphas)} tests)")
            
            if value_alphas:
                value_avg = sum(value_alphas) / len(value_alphas)
                value_dollar_avg = sum(value_dollar_alphas) / len(value_dollar_alphas) if value_dollar_alphas else 0
                print(f"Value Strategy Avg: {value_avg:+.2f}% (${value_dollar_avg:+,.2f}) ({len(value_alphas)} tests)")
    
    print(f"\n{'='*80}")
    print("TIP: Run with --seed <number> for reproducible results")
    print("TIP: Use --threads <number> to control parallel execution (default: 4)")
    print("TIP: Use --optimal-periods to test with strategy-optimal timeframes")
    print("TIP: Increase number of tests for more comprehensive analysis")
    print("\nOptimal Backtesting Periods:")
    print("  MOMENTUM: 3-5 years (captures full market cycles + 200-day SMA warm-up)")
    print("  VALUE: 5-10 years (tests long-term mean reversion across crashes)")
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    exit(main())