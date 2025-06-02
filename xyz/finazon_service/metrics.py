import logging
import pandas as pd
import numpy as np
from datetime import datetime
from config import logger

def pad_to_length(arr, length):
    arr = np.asarray(arr)
    if len(arr) < length:
        pad = np.full(length - len(arr), np.nan)
        return np.concatenate([pad, arr])
    return arr

def calculate_kama(series, window=10, fast=2, slow=30):
    series = np.asarray(series)
    kama = np.zeros(len(series))
    kama[:window] = series[:window]
    fast_alpha = 2 / (fast + 1)
    slow_alpha = 2 / (slow + 1)
    for i in range(window, len(series)):
        change = abs(series[i] - series[i - window])
        volatility = np.sum(np.abs(series[i - window + 1:i + 1] - series[i - window:i]))
        er = change / volatility if volatility != 0 else 0
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama[i] = kama[i - 1] + sc * (series[i] - kama[i - 1])
    return kama

def calculate_mama(series, fast_limit=0.5, slow_limit=0.05):
    mama = np.zeros(len(series))
    fama = np.zeros(len(series))
    mama[0] = series[0]
    fama[0] = series[0]
    for i in range(1, len(series)):
        alpha = fast_limit if abs(series[i] - series[i-1]) > 0 else slow_limit
        mama[i] = alpha * series[i] + (1 - alpha) * mama[i-1]
        fama[i] = 0.5 * alpha * series[i] + (1 - 0.5 * alpha) * fama[i-1]
    return mama, fama

def calculate_t3(series, window=5, vfactor=0.7):
    ema1 = pd.Series(series).ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    ema4 = ema3.ewm(span=window, adjust=False).mean()
    ema5 = ema4.ewm(span=window, adjust=False).mean()
    ema6 = ema5.ewm(span=window, adjust=False).mean()
    c1 = -vfactor ** 3
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    t3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
    return t3.values

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(series).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(series).ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd.values, macd_signal.values, macd_hist.values

def calculate_adx(high, low, close, window=14):
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), high[1:] - high[:-1], 0)
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), low[:-1] - low[1:], 0)
    minus_dm = np.where(minus_dm < 0, 0, minus_dm)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    atr = pd.Series(tr).rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=window).mean()
    adx = np.concatenate(([np.nan], adx))
    return adx

def calculate_adxr(adx, window=14):
    adxr = (adx + np.roll(adx, window)) / 2
    adxr[:window] = np.nan
    return adxr

def calculate_apo(series, fast=12, slow=26):
    ema_fast = pd.Series(series).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(series).ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).values

def calculate_ppo(series, fast=12, slow=26):
    ema_fast = pd.Series(series).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(series).ewm(span=slow, adjust=False).mean()
    return ((ema_fast - ema_slow) / ema_slow * 100).values

def calculate_mom(series, window=10):
    return pd.Series(series).diff(window).values

def calculate_bop(open_, high, low, close):
    # Avoid invalid division by zero
    denominator = high - low
    # Replace zero with NaN to prevent division by zero
    denominator = np.where(denominator == 0, np.nan, denominator)
    bop = (close - open_) / denominator
    bop[(high - low) == 0] = 0
    return bop

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = pd.Series(tp).rolling(window).mean()
    mad = pd.Series(tp).rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci.values

def calculate_cmo(series, window=14):
    diff = pd.Series(series).diff()
    sum_up = diff.where(diff > 0, 0).rolling(window).sum()
    sum_down = -diff.where(diff < 0, 0).rolling(window).sum()
    cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down)
    return cmo.values

def calculate_rocr(series, window=10):
    series = np.array(series)
    rocr = np.full_like(series, fill_value=np.nan, dtype=np.float64)
    rocr[window:] = series[window:] / series[:-window]
    return rocr

def calculate_aroon(high, low, window=14):
    aroon_up = np.zeros(len(high))
    aroon_down = np.zeros(len(low))
    for i in range(window, len(high)):
        window_high = high[i-window+1:i+1]
        window_low = low[i-window+1:i+1]
        aroon_up[i] = 100 * (window_high.argmax() + 1) / window
        aroon_down[i] = 100 * (window_low.argmin() + 1) / window
    return aroon_up, aroon_down

def calculate_aroonosc(aroon_up, aroon_down):
    return aroon_up - aroon_down

def calculate_mfi(high, low, close, volume, window=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), money_flow, 0)
    pos_mf = pd.Series(positive_flow).rolling(window).sum()
    neg_mf = pd.Series(negative_flow).rolling(window).sum()
    mfi = 100 * pos_mf / (pos_mf + neg_mf)
    return mfi.values

def calculate_trix(series, window=15):
    ema1 = pd.Series(series).ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    trix = ema3.pct_change() * 100
    return trix.values

def calculate_ultosc(high, low, close, s1=7, s2=14, s3=28):
    bp = close - np.minimum(low, np.roll(close, 1))
    tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    avg7 = pd.Series(bp).rolling(s1).sum() / pd.Series(tr).rolling(s1).sum()
    avg14 = pd.Series(bp).rolling(s2).sum() / pd.Series(tr).rolling(s2).sum()
    avg28 = pd.Series(bp).rolling(s3).sum() / pd.Series(tr).rolling(s3).sum()
    ultosc = 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
    return ultosc.values


def calculate_mahalanobis_distance(df):
    """Calculate Mahalanobis distance for each point in the dataframe."""
    # Select features for multivariate analysis
    features = df[['close', 'volume', 'rsi', 'macd']].copy()

    # Calculate mean vector and covariance matrix
    mean_vec = features.mean().values
    cov_matrix = features.cov().values

    # Calculate inverse of covariance matrix (handle potential singularity)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Use pseudoinverse if matrix is singular
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # Calculate Mahalanobis distance for each point
    mahalanobis_distances = []
    for i, row in features.iterrows():
        x = row.values
        dist = np.sqrt((x - mean_vec).T.dot(inv_cov_matrix).dot(x - mean_vec))
        mahalanobis_distances.append(dist)

    return np.array(mahalanobis_distances)


def calculate_cusum(df, column='close', k=0.5, h=5, reference_window=20):
    """
    Calculate CUSUM (Cumulative Sum) for anomaly detection.

    Parameters:
    - df: DataFrame containing time series data
    - column: Column to monitor for changes
    - k: Reference value (usually 0.5 standard deviations)
    - h: Decision threshold
    - reference_window: Window size for calculating reference statistics
    """
    series = df[column].values
    cusums = np.zeros(len(series))

    # Calculate mean and standard deviation from reference window
    reference_mean = series[:reference_window].mean()
    reference_std = series[:reference_window].std()

    # Normalize data
    normalized_series = (series - reference_mean) / reference_std

    # Calculate CUSUM
    pos_cusum = 0
    neg_cusum = 0

    for i in range(len(normalized_series)):
        # Positive CUSUM
        pos_cusum = max(0, pos_cusum + normalized_series[i] - k)
        # Negative CUSUM
        neg_cusum = max(0, neg_cusum - normalized_series[i] - k)

        # Store the maximum of positive and negative CUSUM
        cusums[i] = max(pos_cusum, neg_cusum)

        # Reset if threshold is exceeded
        if pos_cusum > h or neg_cusum > h:
            pos_cusum = 0
            neg_cusum = 0

    return cusums


def calculate_cosine_similarity(df, pattern_length=20):
    """
    Calculate cosine similarity between current pattern and historical patterns.

    Parameters:
    - df: DataFrame containing time series data
    - pattern_length: Length of pattern to compare
    """
    close_prices = df['close'].values
    similarities = np.zeros(len(close_prices))

    # Need at least 2*pattern_length data points
    if len(close_prices) < 2 * pattern_length:
        return similarities

    # Use the first pattern_length points as reference
    reference_pattern = close_prices[:pattern_length]

    # Normalize reference pattern
    reference_norm = np.linalg.norm(reference_pattern)
    if reference_norm > 0:
        reference_pattern = reference_pattern / reference_norm

    # Calculate rolling cosine similarity
    for i in range(pattern_length, len(close_prices)):
        current_window = close_prices[i - pattern_length:i]
        current_norm = np.linalg.norm(current_window)

        if current_norm > 0:
            current_window = current_window / current_norm
            similarities[i] = np.dot(reference_pattern, current_window)
        else:
            similarities[i] = 0

    return similarities


def calculate_hurst_exponent(df, column='close', max_lag=100):
    """
    Calculate Hurst exponent to determine if a time series is mean-reverting, random, or trending.

    Parameters:
    - df: DataFrame containing time series data
    - column: Column to analyze
    - max_lag: Maximum lag for calculation
    """
    series = df[column].values
    lags = range(2, min(max_lag, len(series) // 4))

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

    # Calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]  # Hurst exponent is the slope


def calculate_approx_entropy(df, column='close', m=2, r=0.2):
    """
    Calculate approximate entropy, which measures the regularity and unpredictability of fluctuations.

    Parameters:
    - df: DataFrame containing time series data
    - column: Column to analyze
    - m: Length of compared runs of data
    - r: Filtering level, typically 0.2 * std(data)
    """
    series = df[column].values
    N = len(series)

    if N < m + 1:
        return 0

    # Normalize r
    r = r * np.std(series)

    # Create a matrix of all m-length subsequences
    def create_patterns(m_length):
        patterns = np.zeros((N - m_length + 1, m_length))
        for i in range(N - m_length + 1):
            patterns[i] = series[i:i + m_length]
        return patterns

    # Calculate the correlation sum
    def correlation_sum(patterns, m_length):
        N_m = len(patterns)
        count = np.zeros(N_m)

        for i in range(N_m):
            # Calculate distances between pattern i and all other patterns
            distances = np.max(np.abs(patterns - patterns[i]), axis=1)
            # Count patterns within tolerance r
            count[i] = np.sum(distances <= r) / N_m

        return np.sum(np.log(count)) / N_m

    # Calculate for m and m+1
    phi_m = correlation_sum(create_patterns(m), m)
    phi_m_plus_1 = correlation_sum(create_patterns(m + 1), m + 1)

    return phi_m - phi_m_plus_1


def calculate_regime_probability(df, column='close', n_regimes=2, window=100):
    """
    Calculate regime probability using a Hidden Markov Model.

    Parameters:
    - df: DataFrame containing time series data
    - column: Column to analyze
    - n_regimes: Number of regimes/states to identify
    - window: Rolling window size for calculation
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
        # Return placeholder values if hmmlearn is not available
        return np.zeros(len(df))

    series = df[column].values
    returns = np.diff(np.log(series))

    # Reshape for HMM (requires 2D array)
    returns = returns.reshape(-1, 1)

    # Initialize probabilities array
    regime_probs = np.zeros((len(df), n_regimes))

    # Use rolling window to calculate regime probabilities
    for i in range(window, len(returns) + 1):
        # Extract window
        window_data = returns[i - window:i]

        try:
            # Fit HMM model
            model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
            model.fit(window_data)

            # Predict hidden states
            hidden_states = model.predict(window_data)

            # Calculate probability of being in each regime
            last_state = hidden_states[-1]
            regime_probs[i, last_state] = 1.0
        except:
            # In case of convergence issues
            pass

    # For the first window points, use the first calculated probability
    first_valid_prob = next((p for p in regime_probs if np.any(p)), np.zeros(n_regimes))
    regime_probs[:window] = np.tile(first_valid_prob, (window, 1))

    return regime_probs


def compute_batch_metrics(
    df: pd.DataFrame,
    ma_windows: list[int] = (20,),
    vol_window: int = 30,
    rsi_window: int = 14
) -> pd.DataFrame:
    """
    Compute a concise but comprehensive set of indicators for every row of a time-series dataframe.
    All original columns are preserved.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to compute_batch_metrics")

    # Work on a copy to avoid modifying original
    out = df.copy()
    out['timestamp'] = out['timestamp'].astype(int)

    out = out.sort_values("timestamp")



    # --- Basic time stamps ---
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="s")
    out["hour_of_day"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["month_of_year"] = out["datetime"].dt.month

    # --- Returns & volatility ---
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["price_change"] = out["close"].diff()
    out["price_change_pct"] = out["close"].pct_change() * 100
    out["hourly_return"] = out["price_change_pct"]
    out["volatility"] = (
        out["log_return"]
        .rolling(window=vol_window, min_periods=2)
        .std()
        * np.sqrt(252 * 24 * 4)
    )

    # --- Typical price and VWAP ---
    out['typical_price'] = (out['high'] + out['low'] + out['close']) / 3
    out['vwap'] = (out['typical_price'] * out['volume']).cumsum() / out['volume'].cumsum()

    # --- Moving averages ---
    for w in ma_windows:
        out[f"sma_{w}"] = out["close"].rolling(window=w, min_periods=1).mean()
        out[f"ema_{w}"] = out["close"].ewm(span=w, adjust=False).mean()
        # DEMA, TEMA, WMA, TRIMA
        ema = out['close'].ewm(span=w, adjust=False).mean()
        dema = 2 * ema - ema.ewm(span=w, adjust=False).mean()
        out[f'dema_{w}'] = dema
        ema1 = out['close'].ewm(span=w, adjust=False).mean()
        ema2 = ema1.ewm(span=w, adjust=False).mean()
        ema3 = ema2.ewm(span=w, adjust=False).mean()
        out[f'tema_{w}'] = 3 * ema1 - 3 * ema2 + ema3
        weights = np.arange(1, w + 1)
        out[f'wma_{w}'] = out['close'].rolling(window=w).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        out[f'trima_{w}'] = out[f'sma_{w}'].rolling(window=w).mean()


    # Use first window as canonical for some metrics
    main_win = ma_windows[0]
    out["sma"] = out[f"sma_{main_win}"]
    out["ema"] = out[f"ema_{main_win}"]
    out["dema"] = out[f"dema_{main_win}"]
    out["tema"] = out[f"tema_{main_win}"]
    out["wma"] = out[f"wma_{main_win}"]
    out["trima"] = out[f"trima_{main_win}"]

    # --- Volatility metrics (example windows) ---
    out['historical_volatility'] = out['log_return'].rolling(window=30).std() * np.sqrt(252)
    out['realized_volatility'] = out['log_return'].rolling(window=14).std() * np.sqrt(252)

    # --- MACD and relatives ---
    macd, macd_signal, macd_hist = calculate_macd(out['close'])
    out['macd'] = macd
    out['macd_signal'] = macd_signal
    out['macd_hist'] = macd_hist

    # --- Bollinger Bands (using main_win) ---
    rolling_mean = out['close'].rolling(window=main_win).mean()
    rolling_std = out['close'].rolling(window=main_win).std()
    out['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    out['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    out['bollinger_width'] = (out['bollinger_upper'] - out['bollinger_lower']) / rolling_mean

    # --- On-Balance Volume (OBV) ---
    out['obv'] = (np.sign(out['close'].diff()) * out['volume']).fillna(0).cumsum()

    # --- Chaikin Money Flow (CMF) ---
    money_flow_multiplier = ((out['close'] - out['low']) - (out['high'] - out['close'])) / (out['high'] - out['low'])
    money_flow_volume = money_flow_multiplier * out['volume']
    out['cmf'] = money_flow_volume.rolling(window=main_win).sum() / out['volume'].rolling(window=main_win).sum()

    # --- Z-score and EWMA score ---
    out['z_score'] = (out['close'] - out['close'].rolling(window=main_win).mean()) / out['close'].rolling(window=main_win).std()
    ewma = out['close'].ewm(span=main_win).mean()
    out['ewma_score'] = (out['close'] - ewma) / out['close'].rolling(window=main_win).std()

    # --- Sharpe and Sortino ratios (example: 30 window) ---
    returns = out['log_return']
    out['sharpe_ratio'] = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    out['sortino_ratio'] = returns.rolling(window=30).mean() / downside_returns.rolling(window=30).std() * np.sqrt(252)

    # --- Max drawdown ---
    rolling_max = out['close'].rolling(window=30, min_periods=1).max()
    drawdown = (out['close'] / rolling_max - 1.0)
    out['max_drawdown'] = drawdown.rolling(window=30).min()

    # --- Value at Risk (VaR) and Conditional VaR (CVaR) ---
    out['var'] = returns.rolling(window=30).quantile(0.05)
    def rolling_cvar(x):
        var_value = np.percentile(x, 5)
        return x[x <= var_value].mean() if len(x[x <= var_value]) > 0 else var_value
    out['cvar'] = returns.rolling(window=30).apply(rolling_cvar, raw=True)

    # --- ROC (using main_win) ---
    out['roc'] = out['close'].pct_change(periods=main_win) * 100

    # --- RSI (Wilder, use rsi_window) ---
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # --- Stochastic Oscillator, StochRSI, Williams %R (using 14 as standard) ---
    out['stoch_k'] = 100 * ((out['close'] - out['low'].rolling(window=14).min()) /
                            (out['high'].rolling(window=14).max() - out['low'].rolling(window=14).min()))
    out['stoch_d'] = out['stoch_k'].rolling(window=3).mean()
    out['stoch'] = out['stoch_d']
    rsi = out['rsi']
    out['stochrsi'] = 100 * ((rsi - rsi.rolling(window=14).min()) /
                             (rsi.rolling(window=14).max() - rsi.rolling(window=14).min()))
    out['willr'] = -100 * ((out['high'].rolling(window=14).max() - out['close']) /
                           (out['high'].rolling(window=14).max() - out['low'].rolling(window=14).min()))

    # --- Advanced indicators ---
    close = out['close'].values
    open_ = out['open'].values
    high = out['high'].values
    low = out['low'].values
    volume = out['volume'].values
    out['kama'] = calculate_kama(close)
    mama, fama = calculate_mama(close)
    t3 = calculate_t3(close)
    print("check lengths of mama, fama, t3:")
    print(len(out), len(mama), len(fama), len(t3))

    out["mama"] = pad_to_length(mama, len(out))
    out["fama"] = pad_to_length(fama, len(out))
    out["t3"] = pad_to_length(t3, len(out))


    out['adx'] = calculate_adx(high, low, close)
    out['adxr'] = calculate_adxr(out['adx'])
    out['apo'] = calculate_apo(close)
    out['ppo'] = calculate_ppo(close)
    out['mom'] = calculate_mom(close)
    out['bop'] = calculate_bop(open_, high, low, close)
    out['cci'] = calculate_cci(high, low, close)
    out['cmo'] = calculate_cmo(close)
    out['rocr'] = calculate_rocr(close)
    aroon_up, aroon_down = calculate_aroon(high, low)
    out['aroon'] = aroon_up
    out['aroonosc'] = calculate_aroonosc(aroon_up, aroon_down)
    out['mfi'] = calculate_mfi(high, low, close, volume)
    out['trix'] = calculate_trix(close)
    out['ultosc'] = calculate_ultosc(high, low, close)

    # --- Fill NaNs selectively (only for a few metrics) ---
    out['rsi'] = out['rsi'].fillna(50)
    out['adx'] = out['adx'].fillna(0)
    out['macd'] = out['macd'].fillna(0)
    out['macd_signal'] = out['macd_signal'].fillna(0)
    out['bollinger_upper'] = out['bollinger_upper'].fillna(out['close'])
    out['bollinger_lower'] = out['bollinger_lower'].fillna(out['close'])



    # --- Clean up infinities ---
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.debug(f"Compute Batch Metrics Out:\n Timestamp: {out['timestamp']} {out.tail}")
    return out


