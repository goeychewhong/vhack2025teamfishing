import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt
from joblib import Parallel, delayed
import os

def prepare_trading_data(df, 
                        n_future_bars, tp_threshold, sl_threshold, volatility_threshold,split_date='2024-03-25 00:00:00', ):
    """
    Enhanced trading data preparation with:
    - Proper time-series split
    - Volatility-aware signal generation
    - Leakage-proof scaling
    - HMM-ready outputs
    """
    # Create copy and ensure datetime index
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Calculate targets
    df['future_return'] = df['close'].pct_change(n_future_bars).shift(-n_future_bars)
    rolling_vol = df['price_std_7']
    
    # Volatility filter
    vol_filter = rolling_vol > (rolling_vol.mean() * volatility_threshold)
    
    # 3-Class signal logic
    conditions = [
        (df['future_return'] > tp_threshold) & (~vol_filter),   # Long
        (df['future_return'] < sl_threshold) & (~vol_filter),   # Short
        (vol_filter) | (df['future_return'].abs() <= 0.01)      # Neutral
    ]
    
    df['3class_signal'] = np.select(
        condlist=conditions,  
        choicelist=[1, -1, 0],  
        default=0
    )
    
    # Time-based split
    train = df.loc[:split_date]
    test = df.loc[split_date:]
    
    # Define features (exclude targets and metadata)
    feature_cols = [col for col in df.columns if col not in [
        'future_return', '3class_signal', 'start_time', 'close', 'datetime'
    ]]
    
    # Split features and targets
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train['3class_signal'], test['3class_signal']
    
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Prepare HMM data (unscaled)
    train_regime = X_train.copy()
    test_regime = X_test.copy()
    
    # Clean NaN values
    valid_idx = X_train_scaled.index.intersection(y_train.dropna().index)
    X_train_scaled = X_train_scaled.loc[valid_idx]
    y_train = y_train.loc[valid_idx]
    
    return X_train_scaled, X_test_scaled, y_train, y_test, train_regime, test_regime, scaler


def backtest(df, n_future_bars, tp_threshold, sl_threshold, volatility_threshold):
    # Prepare trading data
    X_train, X_test, y_train, y_test, train_regime, test_regime, scaler = prepare_trading_data(
        df.copy(), 
        split_date='2024-03-25 00:00:00', 
        n_future_bars=n_future_bars, 
        tp_threshold=tp_threshold, 
        sl_threshold=sl_threshold, 
        volatility_threshold=volatility_threshold
    )
    
    # Filter data up to the split datetime
    df = df[df.index <= '2024-03-25 00:00:00'].copy()
    df.drop(
        ['exchange_whale_ratio', 'coinbase_premium_index', 'taker_buy_sell_ratio', 
        'netflow', '1_day_lag_coinbase', '7_day_lag_coinbase', 'taker_buy_sell_ratio_sma_7', 
        'taker_buy_sell_ratio_sma_30', 'price_std_7', 'price_std_30'], 
        axis=1, inplace=True
    )
    
    # Combine data
    if len(df) != len(y_train):
        raise ValueError("Length of `df` and `y_train` must match.")
    df['3class_signal'] = y_train.values
    df['daily_pnl'] = 0.0
    df['cumu_pnl'] = 0.0
    df['drawdown'] = 0.0
    df['trade'] = 0
    
    # Reset index to ensure numeric indexing
    df.reset_index(drop=True, inplace=True)
    
    # Calculate trades and PnL
    for i in range(1, len(df)):
        df.loc[i, 'trade'] = abs(df.loc[i, '3class_signal'] - df.loc[i-1, '3class_signal'])
        df.loc[i, 'daily_pnl'] = df.loc[i-1, '3class_signal'] * ((df.loc[i, 'close'] / df.loc[i-1, 'close']) - 1) - 0.0006 * df.loc[i, 'trade']
        df.loc[i, 'cumu_pnl'] = df.loc[i-1, 'cumu_pnl'] + df.loc[i, 'daily_pnl']
        df.loc[i, 'drawdown'] = df.loc[i, 'cumu_pnl'] - df.loc[:i, 'cumu_pnl'].max()
    
    # Calculate metrics
    sharpe = df['daily_pnl'].mean() * sqrt(365) / df['daily_pnl'].std()
    total_trade = df['trade'].sum()
    mdd = df['drawdown'].min()
    
    # Save results to a unique file for each process
    results = [n_future_bars, tp_threshold, sl_threshold, volatility_threshold, sharpe, total_trade, mdd]
    results = pd.DataFrame([results], columns=['n_future_bars', 'tp_threshold', 'sl_threshold', 'volatility_threshold', 'sharpe', 'total_trade', 'mdd'])
    results.to_csv("backtestForSplit.csv", mode='a', header=False)
    
    print("Sharpe ratio:", sharpe, "Total trades:", total_trade, "MDD:", mdd)


# Parallel execution
df = pd.read_csv('data.csv')  # Load the dataset once
df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure 'datetime' is in datetime format
df.set_index('datetime', inplace=True)  # Set 'datetime' as the index

Parallel(n_jobs=os.cpu_count())(delayed(backtest)(df, n_future_bars, tp_threshold, sl_threshold, volatility_threshold) 
                                for n_future_bars in [38,54,64,72,84,96]
                                for tp_threshold in np.arange(0.0, 0.1, 0.02)
                                for sl_threshold in np.arange (0, -0.08, -0.02) 
                                for volatility_threshold in np.arange(2.0, 3.0, 0.5))