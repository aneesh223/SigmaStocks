#!/usr/bin/env python3
"""
Test script to demonstrate the impact of Convolutional Order Book analysis
on trading recommendations.

This script compares recommendations with and without microstructure analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.logic import get_trading_recommendation
from src.microstructure import analyze_liquidity

def test_cob_impact():
    """Test the impact of COB on trading recommendations."""
    
    print("="*80)
    print("CONVOLUTIONAL ORDER BOOK IMPACT TEST")
    print("="*80)
    print()
    print("This test compares trading recommendations WITH and WITHOUT microstructure")
    print("analysis to demonstrate the actual impact of the COB feature.")
    print()
    
    # Create mock sentiment and price data
    sentiment_df = pd.DataFrame({'Compound_Score': [0.5] * 10})
    price_data = pd.DataFrame({
        'Close': [150 + i * 0.1 for i in range(100)]
    }, index=pd.date_range(start='2024-01-01', periods=100, freq='1D'))
    
    # Test with multiple tickers and buy scores
    test_cases = [
        ('AAPL', 6.5, 'momentum'),
        ('TSLA', 7.0, 'momentum'),
        ('MSFT', 5.8, 'value'),
        ('NVDA', 6.2, 'momentum'),
        ('GOOGL', 5.5, 'value'),
    ]
    
    results = []
    
    for ticker, buy_score, strategy in test_cases:
        print(f"Testing {ticker} (Buy Score: {buy_score}, Strategy: {strategy.upper()})")
        print("-"*80)
        
        # First, get the microstructure analysis
        try:
            microstructure = analyze_liquidity(ticker)
            anomaly_score = microstructure['anomaly_score']
            status = microstructure['status']
            confidence_level = microstructure['confidence']
            
            print(f"  Microstructure Analysis:")
            print(f"    - Anomaly Score: {anomaly_score:.3f}")
            print(f"    - Status: {status}")
            print(f"    - Data Confidence: {confidence_level}")
            print()
            
            # Get recommendation WITH microstructure
            result_with = get_trading_recommendation(
                ticker=ticker,
                final_buy_score=buy_score,
                sentiment_df=sentiment_df,
                price_data=price_data,
                strategy=strategy
            )
            
            # Temporarily disable microstructure by mocking it to return neutral
            import unittest.mock as mock
            with mock.patch('src.microstructure.analyze_liquidity') as mock_analyze:
                # Return neutral score with low confidence (will be skipped)
                mock_analyze.return_value = {
                    'anomaly_score': 0.5,
                    'status': 'Normal market conditions',
                    'confidence': 'low'
                }
                
                result_without = get_trading_recommendation(
                    ticker=ticker,
                    final_buy_score=buy_score,
                    sentiment_df=sentiment_df,
                    price_data=price_data,
                    strategy=strategy
                )
            
            # Compare results
            confidence_diff = result_with['confidence'] - result_without['confidence']
            microstructure_applied = result_with.get('microstructure_applied', False)
            
            print(f"  Results:")
            print(f"    WITHOUT Microstructure:")
            print(f"      - Recommendation: {result_without['recommendation']}")
            print(f"      - Confidence: {result_without['confidence']:.1f}%")
            print()
            print(f"    WITH Microstructure:")
            print(f"      - Recommendation: {result_with['recommendation']}")
            print(f"      - Confidence: {result_with['confidence']:.1f}%")
            print(f"      - Applied: {microstructure_applied}")
            print()
            
            if microstructure_applied and abs(confidence_diff) > 0.01:
                change_direction = "increased" if confidence_diff > 0 else "decreased"
                change_pct = abs(confidence_diff)
                print(f"  ✅ IMPACT DETECTED: Confidence {change_direction} by {change_pct:.1f}%")
                
                # Explain why
                if anomaly_score > 0.85:
                    if result_with['recommendation'] == 'BUY':
                        print(f"     Reason: Critical liquidity void (score {anomaly_score:.3f}) reduced BUY confidence")
                    else:
                        print(f"     Reason: Critical liquidity void (score {anomaly_score:.3f}) confirmed SELL signal")
                elif anomaly_score > 0.70:
                    print(f"     Reason: High microstructure volatility (score {anomaly_score:.3f}) reduced confidence")
                elif anomaly_score < 0.20:
                    print(f"     Reason: Strong accumulation zone (score {anomaly_score:.3f}) boosted confidence")
            elif microstructure_applied:
                print(f"  ℹ️  Microstructure applied but no significant change (anomaly score in normal range)")
            else:
                print(f"  ⚠️  Microstructure not applied (insufficient data or low confidence)")
            
            results.append({
                'ticker': ticker,
                'buy_score': buy_score,
                'strategy': strategy,
                'anomaly_score': anomaly_score,
                'confidence_without': result_without['confidence'],
                'confidence_with': result_with['confidence'],
                'difference': confidence_diff,
                'applied': microstructure_applied
            })
            
        except Exception as e:
            print(f"  ❌ Error testing {ticker}: {e}")
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    applied_count = sum(1 for r in results if r['applied'])
    changed_count = sum(1 for r in results if r['applied'] and abs(r['difference']) > 0.01)
    
    print(f"Total Tests: {len(results)}")
    print(f"Microstructure Applied: {applied_count}/{len(results)}")
    print(f"Confidence Changed: {changed_count}/{len(results)}")
    print()
    
    if changed_count > 0:
        print("Detailed Changes:")
        for r in results:
            if r['applied'] and abs(r['difference']) > 0.01:
                direction = "↑" if r['difference'] > 0 else "↓"
                print(f"  {r['ticker']:6s} | Anomaly: {r['anomaly_score']:.3f} | "
                      f"Confidence: {r['confidence_without']:.1f}% → {r['confidence_with']:.1f}% "
                      f"({direction} {abs(r['difference']):.1f}%)")
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    
    if changed_count > 0:
        print(f"✅ COB feature IS actively changing trading recommendations!")
        print(f"   {changed_count} out of {len(results)} tests showed confidence adjustments.")
    elif applied_count > 0:
        print(f"ℹ️  COB feature is running but anomaly scores are in normal range.")
        print(f"   No significant adjustments needed for current market conditions.")
    else:
        print(f"⚠️  COB feature could not be applied (insufficient intraday data).")
        print(f"   This is expected if market is closed or data is unavailable.")
    
    print()

if __name__ == "__main__":
    test_cob_impact()
