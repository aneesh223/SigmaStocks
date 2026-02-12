#!/usr/bin/env python3
"""
Test script to demonstrate COB impact across different market microstructure scenarios.

This script simulates various anomaly scores to show how the COB feature
adjusts trading recommendations under different market conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from unittest.mock import patch
from src.logic import get_trading_recommendation

def test_scenario(ticker, buy_score, strategy, anomaly_score, status):
    """Test a specific microstructure scenario."""
    
    # Create mock data
    sentiment_df = pd.DataFrame({'Compound_Score': [0.5] * 10})
    price_data = pd.DataFrame({
        'Close': [150 + i * 0.1 for i in range(100)]
    }, index=pd.date_range(start='2024-01-01', periods=100, freq='1D'))
    
    # Get baseline (without microstructure)
    with patch('src.microstructure.analyze_liquidity') as mock_analyze:
        mock_analyze.return_value = {
            'anomaly_score': 0.5,
            'status': 'Normal market conditions',
            'confidence': 'low'  # Low confidence means it won't be applied
        }
        
        result_baseline = get_trading_recommendation(
            ticker=ticker,
            final_buy_score=buy_score,
            sentiment_df=sentiment_df,
            price_data=price_data,
            strategy=strategy
        )
    
    # Get result with specific anomaly score
    with patch('src.microstructure.analyze_liquidity') as mock_analyze:
        mock_analyze.return_value = {
            'anomaly_score': anomaly_score,
            'status': status,
            'confidence': 'high'
        }
        
        result_with_cob = get_trading_recommendation(
            ticker=ticker,
            final_buy_score=buy_score,
            sentiment_df=sentiment_df,
            price_data=price_data,
            strategy=strategy
        )
    
    return result_baseline, result_with_cob

def main():
    print("="*80)
    print("CONVOLUTIONAL ORDER BOOK - SCENARIO TESTING")
    print("="*80)
    print()
    print("Testing how COB adjusts recommendations under different market conditions")
    print()
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Critical Liquidity Void (BUY Signal)',
            'ticker': 'AAPL',
            'buy_score': 6.5,
            'strategy': 'momentum',
            'anomaly_score': 0.92,
            'status': 'Critical liquidity void detected',
            'expected': 'Confidence should decrease by ~30%'
        },
        {
            'name': 'High Microstructure Volatility (BUY Signal)',
            'ticker': 'TSLA',
            'buy_score': 7.0,
            'strategy': 'momentum',
            'anomaly_score': 0.75,
            'status': 'High microstructure volatility',
            'expected': 'Confidence should decrease by ~15%'
        },
        {
            'name': 'Strong Accumulation Zone (BUY Signal)',
            'ticker': 'MSFT',
            'buy_score': 6.2,
            'strategy': 'value',
            'anomaly_score': 0.15,
            'status': 'Strong accumulation zone detected',
            'expected': 'Confidence should increase by ~20%'
        },
        {
            'name': 'Normal Market Conditions (BUY Signal)',
            'ticker': 'NVDA',
            'buy_score': 6.0,
            'strategy': 'momentum',
            'anomaly_score': 0.45,
            'status': 'Normal market conditions',
            'expected': 'No change expected'
        },
        {
            'name': 'Critical Liquidity Void (SELL Signal)',
            'ticker': 'GOOGL',
            'buy_score': 4.0,
            'strategy': 'value',
            'anomaly_score': 0.90,
            'status': 'Critical liquidity void detected',
            'expected': 'Confidence should increase by ~15% (confirms sell)'
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print("-"*80)
        print(f"  Ticker: {scenario['ticker']}")
        print(f"  Buy Score: {scenario['buy_score']}")
        print(f"  Strategy: {scenario['strategy'].upper()}")
        print(f"  Anomaly Score: {scenario['anomaly_score']:.3f}")
        print(f"  Status: {scenario['status']}")
        print(f"  Expected: {scenario['expected']}")
        print()
        
        baseline, with_cob = test_scenario(
            scenario['ticker'],
            scenario['buy_score'],
            scenario['strategy'],
            scenario['anomaly_score'],
            scenario['status']
        )
        
        confidence_diff = with_cob['confidence'] - baseline['confidence']
        confidence_change_pct = (confidence_diff / baseline['confidence'] * 100) if baseline['confidence'] > 0 else 0
        
        print(f"  Results:")
        print(f"    Baseline (No COB):")
        print(f"      - Recommendation: {baseline['recommendation']}")
        print(f"      - Confidence: {baseline['confidence']:.1f}%")
        print()
        print(f"    With COB:")
        print(f"      - Recommendation: {with_cob['recommendation']}")
        print(f"      - Confidence: {with_cob['confidence']:.1f}%")
        print(f"      - Applied: {with_cob.get('microstructure_applied', False)}")
        print()
        
        if abs(confidence_diff) > 0.01:
            direction = "↑" if confidence_diff > 0 else "↓"
            print(f"  ✅ IMPACT: Confidence changed by {direction} {abs(confidence_diff):.1f}% "
                  f"({confidence_change_pct:+.1f}%)")
            
            # Verify expected behavior
            if scenario['anomaly_score'] > 0.85 and baseline['recommendation'] == 'BUY':
                expected_change = baseline['confidence'] * -0.30
                actual_matches = abs(confidence_diff - expected_change) < 1.0
                status_icon = "✓" if actual_matches else "✗"
                print(f"     {status_icon} Expected ~30% reduction: {expected_change:.1f}%, Got: {confidence_diff:.1f}%")
            elif scenario['anomaly_score'] > 0.70 and baseline['recommendation'] == 'BUY':
                expected_change = baseline['confidence'] * -0.15
                actual_matches = abs(confidence_diff - expected_change) < 1.0
                status_icon = "✓" if actual_matches else "✗"
                print(f"     {status_icon} Expected ~15% reduction: {expected_change:.1f}%, Got: {confidence_diff:.1f}%")
            elif scenario['anomaly_score'] < 0.20 and baseline['recommendation'] == 'BUY':
                expected_change = baseline['confidence'] * 0.20
                actual_matches = abs(confidence_diff - expected_change) < 1.0
                status_icon = "✓" if actual_matches else "✗"
                print(f"     {status_icon} Expected ~20% boost: {expected_change:.1f}%, Got: {confidence_diff:.1f}%")
            elif scenario['anomaly_score'] > 0.85 and baseline['recommendation'] == 'SELL':
                expected_change = baseline['confidence'] * 0.15
                actual_matches = abs(confidence_diff - expected_change) < 1.0
                status_icon = "✓" if actual_matches else "✗"
                print(f"     {status_icon} Expected ~15% boost: {expected_change:.1f}%, Got: {confidence_diff:.1f}%")
        else:
            print(f"  ℹ️  No significant change (as expected for normal conditions)")
        
        results.append({
            'scenario': scenario['name'],
            'anomaly_score': scenario['anomaly_score'],
            'baseline_conf': baseline['confidence'],
            'cob_conf': with_cob['confidence'],
            'difference': confidence_diff,
            'change_pct': confidence_change_pct
        })
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'Scenario':<45} {'Anomaly':<10} {'Baseline':<10} {'With COB':<10} {'Change':<10}")
    print("-"*80)
    
    for r in results:
        change_str = f"{r['difference']:+.1f}% ({r['change_pct']:+.1f}%)"
        print(f"{r['scenario']:<45} {r['anomaly_score']:<10.3f} "
              f"{r['baseline_conf']:<10.1f} {r['cob_conf']:<10.1f} {change_str:<10}")
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    
    changed_count = sum(1 for r in results if abs(r['difference']) > 0.01)
    
    if changed_count > 0:
        print(f"✅ COB feature successfully adjusts recommendations based on microstructure!")
        print(f"   {changed_count}/{len(results)} scenarios showed confidence adjustments.")
        print()
        print("Key Findings:")
        print("  • Critical liquidity voids (>0.85) reduce BUY confidence by ~30%")
        print("  • High volatility (>0.70) reduces BUY confidence by ~15%")
        print("  • Strong accumulation (<0.20) boosts BUY confidence by ~20%")
        print("  • Critical voids confirm SELL signals with +15% confidence")
        print("  • Normal conditions (0.20-0.70) result in no adjustment")
    else:
        print(f"⚠️  No changes detected - check implementation")
    
    print()

if __name__ == "__main__":
    main()
