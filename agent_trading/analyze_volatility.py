import pandas as pd
import numpy as np

paths = ['data/VNM.csv', 'data/FPT.csv', 'data/VIC.csv', 'data/HPG.csv']

for p in paths:
    df = pd.read_csv(p)
    c = df['close']
    daily_ret = c.pct_change().dropna()
    
    # ATR-like measure
    if 'high' in df.columns and 'low' in df.columns:
        atr_pct = ((df['high'] - df['low']) / c).mean() * 100
    else:
        atr_pct = 0
    
    print(f"\n=== {p} ===")
    print(f"  Rows: {len(df)}")
    print(f"  Close range: {c.min():.1f} - {c.max():.1f}")
    print(f"  Avg daily return: {daily_ret.mean()*100:.3f}%")
    print(f"  Daily volatility (std): {daily_ret.std()*100:.2f}%")
    print(f"  Avg ATR%: {atr_pct:.2f}%")
    print(f"  Max single-day gain: {daily_ret.max()*100:.2f}%")
    print(f"  Max single-day loss: {daily_ret.min()*100:.2f}%")
    
    # How often does price move >4% in 2 days?
    ret_2d = c.pct_change(2).dropna()
    pct_exceed_4 = (ret_2d.abs() > 0.04).mean() * 100
    pct_exceed_5 = (ret_2d.abs() > 0.05).mean() * 100
    pct_exceed_7 = (ret_2d.abs() > 0.07).mean() * 100
    print(f"  2-day move > 4%: {pct_exceed_4:.1f}% of time")
    print(f"  2-day move > 5%: {pct_exceed_5:.1f}% of time")
    print(f"  2-day move > 7%: {pct_exceed_7:.1f}% of time")
    
    # ATR(14) in percentage terms
    if 'atr_14' in df.columns:
        atr14_pct = (df['atr_14'] / c).dropna().mean() * 100
        print(f"  ATR(14) as % of close: {atr14_pct:.2f}%")
