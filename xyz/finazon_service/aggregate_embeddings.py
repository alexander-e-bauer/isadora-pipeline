import pandas as pd
import numpy as np
import traceback

from datetime import datetime, date, timedelta
from config import logger
from xyz.finazon_service.sql_service import (
    Session, Ticker, HistoricalData, ComputedMetrics,
    MarketEmbHour, MarketEmbDay, MarketEmbWeek
)
from api_service import get_ticker_news, extract_article_data, create_news_summary_for_period
from market_analysis import (
    classify_trend_strength, classify_volatility_regime, classify_momentum_phase,
    identify_technical_signals, assess_risk_level
)
from xyz.llm.embedding_generator import get_embedding, num_tokens

# --------- Helper: Aggregate Computed Metrics ---------

import traceback
import pandas as pd


def to_yyyy_mm_dd(dt):
    """
    Converts a datetime, date, or datetime-like string to 'YYYY-MM-DD' format.
    """
    if isinstance(dt, datetime) or isinstance(dt, date):
        return dt.strftime('%Y-%m-%d')
    elif isinstance(dt, str):
        # Try to parse common datetime string formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(dt, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        raise ValueError(f"Unknown date format: {dt}")
    else:
        raise TypeError(f"Unsupported type for date conversion: {type(dt)}")


def fetch_and_summarize_weekly_articles(ticker_symbol, week_start, week_end):
    """
    For a given ticker and week, fetches daily news summaries and summarizes them into a weekly summary.
    Returns a string summary for the week.
    """
    daily_summaries = []
    current_day = week_start
    while current_day < week_end:
        day_str = current_day.strftime('%Y-%m-%d')
        # Fetch and summarize news for the day
        summary = fetch_news_for_period(
            ticker_symbol,
            period_start=day_str,
            period_end=day_str
        )
        if summary and summary.strip():
            daily_summaries.append(summary)
        current_day += timedelta(days=1)
    if not daily_summaries:
        return "No news for this week."
    # Summarize the concatenated daily summaries into a weekly summary
    weekly_summary = create_news_summary_for_period([{"title": "", "article_url": "", "insights": "", "publisher": "", "text": s} for s in daily_summaries], 5120)
    return weekly_summary


def aggregate_metrics(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    try:
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        df.index.name = 'datetime'
        agg_dict = {
            'close': lambda x: x.iloc[-1] if len(x)>0 else np.nan,  # Last closing price
            'open': lambda x: x.iloc[0] if len(x)>0 else np.nan,  # First opening price
            'high': 'max',  # Highest price
            'low': 'min',  # Lowest price
            'volume': 'sum',  # Total volume over the period

            # Moving averages
            'sma': 'mean', 'ema': 'mean', 'dema': 'mean', 'tema': 'mean',
            'wma': 'mean', 'trima': 'mean', 'kama': 'mean', 'mama': 'mean', 't3': 'mean',
            'sma_20': 'mean', 'ema_20': 'mean', 'dema_20': 'mean', 'tema_20': 'mean',
            'wma_20': 'mean', 'trima_20': 'mean',

            # Returns and price behavior
            'log_return': 'mean',
            'historical_volatility': 'mean',
            'realized_volatility': 'mean',
            'hourly_return': 'mean',
            'typical_price': 'mean',
            'price_change': 'mean',
            'price_change_pct': 'mean',

            # Oscillators and indicators
            'macd': 'mean', 'macd_signal': 'mean', 'macd_hist': 'mean',
            'rsi': 'mean', 'stoch': 'mean', 'stochrsi': 'mean', 'stoch_k': 'mean', 'stoch_d': 'mean',
            'willr': 'mean', 'adx': 'mean', 'adxr': 'mean', 'apo': 'mean', 'ppo': 'mean', 'mom': 'mean',
            'bop': 'mean', 'cci': 'mean', 'cmo': 'mean', 'roc': 'mean', 'rocr': 'mean', 'aroon': 'mean',
            'aroonosc': 'mean', 'mfi': 'mean', 'trix': 'mean', 'ultosc': 'mean',

            # Bollinger Bands
            'bollinger_upper': 'mean',
            'bollinger_lower': 'mean',
            'bollinger_width': 'mean',

            # Metrics and analysis scores
            'obv': 'sum',
            'cmf': 'mean',
            'z_score': 'mean',
            'ewma_score': 'mean',
            'sharpe_ratio': 'mean',
            'sortino_ratio': 'mean',
            'max_drawdown': 'min',
            'var': 'mean',
            'cvar': 'mean',

            # Additional indicators
            'fama': 'mean'
        }

        agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}
        agg_df = df.resample(freq).agg(agg_dict_filtered)
        agg_df = agg_df.dropna(subset=['open', 'close'])

        print("agg_df type:", type(agg_df))
        if isinstance(agg_df, pd.Series):
            agg_df = agg_df.to_frame()
        agg_df = agg_df.reset_index()
        return agg_df

    except Exception as e:
        print("An error occurred during aggregation:")
        traceback.print_exc()
        print("df.columns:", df.columns)
        print("df.index.name:", df.index.name)
        raise  # Optionally re-raise the error if you want it to propagate



# --------- Helper: Market State Construction ---------

def make_market_state(row):
    # row: a dict or pd.Series with all needed metrics
    state = {
        'trend_strength': classify_trend_strength(row),
        'volatility_regime': classify_volatility_regime(row),
        'momentum_phase': classify_momentum_phase(row),
        'technical_signals': identify_technical_signals(row),
        'market_position': 'N/A',  # You can add your own logic
        'risk_level': assess_risk_level(row),
        'market_summary': make_market_summary(row)
    }
    return state

def make_market_summary(row):
    # Simple summary, can be made fancier
    return (
        f"Trend: {classify_trend_strength(row)}, "
        f"Volatility: {classify_volatility_regime(row)}, "
        f"Momentum: {classify_momentum_phase(row)}, "
        f"Signals: {identify_technical_signals(row)}, "
        f"Risk: {assess_risk_level(row)}"
    )

# --------- Helper: News Fetching ---------
def fetch_news_for_period(ticker_symbol, period_start, period_end):
    # Example: Fetch news for Apple (AAPL) with a date filter
    #from_date = "2025-05-20"  # Articles published on or after this date
    #to_date = "2025-05-21"  # Articles published on or before this date
    articles = get_ticker_news(ticker_symbol, limit=24, published_from=period_start, published_to=period_end)
    extracted_data = extract_article_data(articles)
    print(extracted_data)
    text = create_news_summary_for_period(extracted_data, 128)
    return text

# --------- Main Pipeline ---------

def process_aggregated_embeddings(ticker_symbol, session=None):
    """
    Aggregates computed metrics into hourly/daily/weekly, generates semantic state,
    fetches news, creates embeddings, and stores in the market_emb_* tables.
    """
    if session is None:
        session = Session()

    ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
    if not ticker:
        print(f"Ticker {ticker_symbol} not found.")
        return

    # 1. Fetch computed metrics and historical data as DataFrame
    q = (
        session.query(
            HistoricalData.timestamp,
            HistoricalData.open,
            HistoricalData.close,
            HistoricalData.high,
            HistoricalData.low,
            HistoricalData.volume,
            # Moving averages
            ComputedMetrics.sma,
            ComputedMetrics.ema,
            ComputedMetrics.dema,
            ComputedMetrics.tema,
            ComputedMetrics.wma,
            ComputedMetrics.trima,
            ComputedMetrics.kama,
            ComputedMetrics.mama,
            ComputedMetrics.t3,
            ComputedMetrics.sma_20,
            ComputedMetrics.ema_20,
            ComputedMetrics.dema_20,
            ComputedMetrics.tema_20,
            ComputedMetrics.wma_20,
            ComputedMetrics.trima_20,

            # Returns and price behavior
            ComputedMetrics.log_return,
            ComputedMetrics.hourly_return,
            ComputedMetrics.historical_volatility,
            ComputedMetrics.realized_volatility,
            ComputedMetrics.typical_price,
            ComputedMetrics.price_change,
            ComputedMetrics.price_change_pct,

            # Oscillators and indicators
            ComputedMetrics.macd,
            ComputedMetrics.macd_signal,
            ComputedMetrics.macd_hist,
            ComputedMetrics.rsi,
            ComputedMetrics.stoch,
            ComputedMetrics.stochrsi,
            ComputedMetrics.stoch_k,
            ComputedMetrics.stoch_d,
            ComputedMetrics.willr,
            ComputedMetrics.adx,
            ComputedMetrics.adxr,
            ComputedMetrics.apo,
            ComputedMetrics.ppo,
            ComputedMetrics.mom,
            ComputedMetrics.bop,
            ComputedMetrics.cci,
            ComputedMetrics.cmo,
            ComputedMetrics.roc,
            ComputedMetrics.rocr,
            ComputedMetrics.aroon,
            ComputedMetrics.aroonosc,
            ComputedMetrics.mfi,
            ComputedMetrics.trix,
            ComputedMetrics.ultosc,

            # Bollinger Bands
            ComputedMetrics.bollinger_upper,
            ComputedMetrics.bollinger_lower,
            ComputedMetrics.bollinger_width,

            # Metrics and analysis scores
            ComputedMetrics.obv,
            ComputedMetrics.cmf,
            ComputedMetrics.z_score,
            ComputedMetrics.ewma_score,
            ComputedMetrics.sharpe_ratio,
            ComputedMetrics.sortino_ratio,
            ComputedMetrics.max_drawdown,
            ComputedMetrics.var,
            ComputedMetrics.cvar,

            # Additional indicators
            ComputedMetrics.fama
        )
        .join(ComputedMetrics, ComputedMetrics.historical_data_id == HistoricalData.id)
        .filter(HistoricalData.ticker_id == ticker.id)
        .order_by(HistoricalData.timestamp)
    )

    df = pd.DataFrame(q.all(), columns=[col['name'] for col in q.column_descriptions])

    if df.empty:
        print(f"No computed metrics found for {ticker_symbol}.")
        return

    # 2. For each granularity
    for freq, Table in [
        ('h', MarketEmbHour),
        ('d', MarketEmbDay),
        ('W', MarketEmbWeek)
    ]:
        agg_df = aggregate_metrics(df, freq)
        for i, row in agg_df.iterrows():
            row_dict = row.to_dict()
            # 3. Semantic market state
            state = make_market_state(row_dict)
            # 4. News
            period_start = row['datetime']

            exists = (
                session.query(Table)
                .filter_by(ticker_id=ticker.id, period_start=period_start)
                .first()
            )
            if exists:
                logger.debug("Skipping already embedded data")
                continue  # Skip already embedded period
            if freq == 'h':
                period_end = period_start + timedelta(hours=1)
                news_blob = "None"
            elif freq == 'd':
                period_end = period_start + timedelta(days=1)
                news_blob = fetch_news_for_period(ticker_symbol, to_yyyy_mm_dd(period_start), to_yyyy_mm_dd(period_end))
            elif freq == 'W':
                period_end = period_start + timedelta(weeks=1)
                week_start = period_start.date()
                week_end = period_end.date()
                news_blob = fetch_and_summarize_weekly_articles(
                    ticker_symbol,
                    week_start=week_start,
                    week_end=week_end
                )

            else:
                logger.error("Incorrect data storage frequency")
                continue

            print(f"blob = {news_blob}")
            # 5. Embedding (text = summary + news)
            text_to_embed = f"{state['market_summary']}\nNews: {news_blob}"
            num_tokens(text_to_embed)
            embedding_vector = get_embedding(text_to_embed)
            # 6. Store in DB
            emb = Table(
                ticker_id=ticker.id,
                period_start=period_start,
                period_end=period_end,
                trend_strength=state['trend_strength'],
                volatility_regime=state['volatility_regime'],
                momentum_phase=state['momentum_phase'],
                technical_signals=state['technical_signals'],
                market_position=state['market_position'],
                risk_level=state['risk_level'],
                market_summary=state['market_summary'],
                news_headlines=news_blob,
                embedding_vector=embedding_vector
            )
            session.add(emb)
        session.commit()
    print(f"Aggregated embeddings for {ticker_symbol} created successfully.")

# ---------- Example Usage ----------
# process_aggregated_embeddings('AAPL')  # Call this after updating time series data
if __name__ == "__main__":
    # Example: Fetch news for Apple (AAPL) with a date filter
    ticker_symbol = "AAPL"
    from_date = "2025-05-20"  # Articles published on or after this date
    to_date = "2025-05-21"  # Articles published on or before this date
    news = fetch_news_for_period(ticker_symbol, from_date, to_date)
    print(news)
