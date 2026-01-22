#!/usr/bin/env python3
"""
Comprehensive Algorithm Testing
Tests across multiple time periods, market conditions, and sectors
"""

import subprocess
import sys
import os
from datetime import datetime

def run_backtest(ticker, strategy, start_date, end_date, description=""):
    """Run a single backtest and capture results"""
    cmd = ['python3', 'backtester/run_backtest.py', ticker, strategy, start_date, end_date, '--no-plot']
    
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {ticker} {strategy.upper()} | {start_date} to {end_date}")
    if description:
        print(f"📝 {description}")
    print('='*80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            strategy_return = None
            buy_hold_return = None
            alpha = None
            trades = None
            regime = None
            
            for line in lines:
                if "📈 Strategy Return:" in line:
                    strategy_return = line.split(":")[1].strip()
                elif "📊 Buy & Hold Return:" in line:
                    buy_hold_return = line.split(":")[1].strip()
                elif "🎯 Alpha (Outperformance):" in line:
                    alpha = line.split(":")[1].strip()
                elif "🔄 Number of Trades:" in line:
                    trades = line.split(":")[1].strip()
                elif "Initial Regime:" in line:
                    regime = line.split(":")[1].strip()
            
            return {
                'success': True,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'alpha': alpha,
                'trades': trades,
                'regime': regime,
                'ticker': ticker,
                'strategy': strategy.upper(),
                'period': f"{start_date} to {end_date}",
                'description': description
            }
        else:
            print(f"❌ FAILED: {result.stderr}")
            return {'success': False, 'ticker': ticker, 'strategy': strategy.upper(), 'error': result.stderr}
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {'success': False, 'ticker': ticker, 'strategy': strategy.upper(), 'error': str(e)}

def main():
    """Run comprehensive tests"""
    
    print("🚀 COMPREHENSIVE ALGORITHM TESTING")
    print("Testing across multiple time periods, market conditions, and sectors")
    print("=" * 80)
    
    # Test configurations: (ticker, strategy, start_date, end_date, description)
    test_configs = [
        # BULL MARKETS - Different sectors
        ('TSLA', 'm', '2023-01-01', '2023-03-01', 'BULL MARKET: Tesla momentum in strong bull run'),
        ('NVDA', 'm', '2023-05-01', '2023-08-01', 'STRONG BULL: NVIDIA AI boom momentum'),
        ('AAPL', 'm', '2023-10-01', '2023-12-01', 'BULL MARKET: Apple momentum in tech rally'),
        ('MSFT', 'm', '2023-01-01', '2023-04-01', 'BULL MARKET: Microsoft momentum in AI hype'),
        
        # VALUE STRATEGIES - Different conditions
        ('AAPL', 'v', '2023-03-01', '2023-06-01', 'VALUE: Apple during sideways consolidation'),
        ('MSFT', 'v', '2023-01-01', '2023-03-01', 'VALUE: Microsoft during mixed conditions'),
        ('TSLA', 'v', '2023-06-01', '2023-09-01', 'VALUE: Tesla during volatile summer'),
        
        # DIFFERENT SECTORS
        ('JPM', 'm', '2023-01-01', '2023-04-01', 'FINANCIALS: JPMorgan momentum strategy'),
        ('XOM', 'v', '2023-01-01', '2023-04-01', 'ENERGY: Exxon value strategy'),
        ('JNJ', 'v', '2023-01-01', '2023-04-01', 'HEALTHCARE: J&J defensive value'),
        
        # DIFFERENT TIME PERIODS
        ('TSLA', 'm', '2023-02-01', '2023-04-01', 'SHORT TERM: 2-month Tesla momentum'),
        ('AAPL', 'm', '2023-01-01', '2023-06-01', 'MEDIUM TERM: 5-month Apple momentum'),
        ('NVDA', 'v', '2023-03-01', '2023-08-01', 'LONG TERM: 5-month NVIDIA value'),
        
        # VOLATILE CONDITIONS
        ('TSLA', 'm', '2023-04-01', '2023-07-01', 'HIGH VOLATILITY: Tesla spring volatility'),
        ('NVDA', 'm', '2023-08-01', '2023-11-01', 'POST-BOOM: NVIDIA after AI peak'),
        
        # MIXED CONDITIONS
        ('AAPL', 'v', '2023-07-01', '2023-10-01', 'MIXED: Apple summer to fall transition'),
        ('MSFT', 'm', '2023-06-01', '2023-09-01', 'MIXED: Microsoft summer momentum'),
    ]
    
    # For quick testing, uncomment the line below to run just one test:
    # test_configs = [('TSLA', 'm', '2023-01-01', '2023-03-01', 'QUICK TEST: Tesla momentum')]
    
    results = []
    successful = 0
    
    for ticker, strategy, start_date, end_date, description in test_configs:
        result = run_backtest(ticker, strategy, start_date, end_date, description)
        results.append(result)
        if result['success']:
            successful += 1
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*80}")
    print(f"✅ Successful Tests: {successful}/{len(test_configs)}")
    print()
    
    # Group results by category
    bull_markets = []
    value_strategies = []
    sectors = []
    volatility_tests = []
    
    for result in results:
        if result['success']:
            if 'BULL' in result['description']:
                bull_markets.append(result)
            elif 'VALUE' in result['description']:
                value_strategies.append(result)
            elif any(sector in result['description'] for sector in ['FINANCIALS', 'ENERGY', 'HEALTHCARE']):
                sectors.append(result)
            elif 'VOLATILITY' in result['description'] or 'MIXED' in result['description']:
                volatility_tests.append(result)
    
    # Print category summaries
    categories = [
        ("🚀 BULL MARKET PERFORMANCE", bull_markets),
        ("💎 VALUE STRATEGY PERFORMANCE", value_strategies), 
        ("🏭 SECTOR DIVERSIFICATION", sectors),
        ("⚡ VOLATILITY & MIXED CONDITIONS", volatility_tests)
    ]
    
    for category_name, category_results in categories:
        if category_results:
            print(f"\n{category_name}")
            print("-" * 60)
            for result in category_results:
                alpha_color = "🟢" if result['alpha'] and float(result['alpha'].replace('%', '').replace('+', '')) > 0 else "🔴" if result['alpha'] and float(result['alpha'].replace('%', '').replace('+', '')) < -5 else "🟡"
                print(f"{alpha_color} {result['ticker']} {result['strategy']}: {result['alpha']} alpha | {result['trades']} trades | {result['regime']}")
                print(f"   📈 Strategy: {result['strategy_return']} vs Buy-Hold: {result['buy_hold_return']}")
                print(f"   📝 {result['description']}")
                print()
    
    # Calculate overall statistics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        alphas = []
        for result in successful_results:
            if result['alpha']:
                try:
                    alpha_val = float(result['alpha'].replace('%', '').replace('+', ''))
                    alphas.append(alpha_val)
                except:
                    pass
        
        if alphas:
            avg_alpha = sum(alphas) / len(alphas)
            positive_alpha_count = sum(1 for a in alphas if a > 0)
            near_zero_count = sum(1 for a in alphas if -2 <= a <= 2)  # Within 2% of buy-hold
            
            print(f"\n📈 OVERALL STATISTICS")
            print("-" * 40)
            print(f"Average Alpha: {avg_alpha:+.2f}%")
            print(f"Positive Alpha: {positive_alpha_count}/{len(alphas)} ({positive_alpha_count/len(alphas)*100:.1f}%)")
            print(f"Near Buy-Hold (±2%): {near_zero_count}/{len(alphas)} ({near_zero_count/len(alphas)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    
    return 0 if successful == len(test_configs) else 1

if __name__ == "__main__":
    exit(main())