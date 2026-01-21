# SigmaStocks Backtesting - Quick Reference

## Basic Usage
```bash
python backtest.py <ticker> <strategy> <period> [cash] [--plot] [--save]
```

## Examples
```bash
# Quick test
python backtest.py TSLA m 2023-01-20 2023-07-20

# With custom capital and charts
python backtest.py AAPL v 90 25000 --plot --save

# See all examples
python examples.py

# Batch testing
python batch_backtest.py
```

## Parameters
- **ticker**: TSLA, AAPL, NVDA, etc.
- **strategy**: `m` (momentum) or `v` (value)
- **period**: Days (30, 90, 180) OR dates (2023-01-20 2023-07-20)
- **cash**: Starting capital (default: $10,000)
- **--plot**: Show charts
- **--save**: Save to JSON

## Tips
- Use historical dates (15+ days old) to avoid API limits
- Momentum: 30-90 day periods work best
- Value: 90-180 day periods work best
- Test multiple tickers for validation