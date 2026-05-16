import pandas as pd
import numpy as np
import traceback
import tiktoken
from datetime import datetime, date, timedelta, time as dt_time
import pytz
from config import logger, OAI
from xyz.finazon_service.api_service import (
    create_weekly_summary_for_period, extract_article_data,
    create_news_summary_for_period, filter_articles, scrape_cache,
    get_ticker_news_polygon,
    fetch_news_for_period_with_flags,
    fetch_and_summarize_weekly_articles_cached_with_flags,
    TimelinePinManager, parse_news_flags
)
from xyz.finazon_service.sql_service import (
    get_db_session, Ticker, HistoricalData, ComputedMetrics,
    MarketEmbThirtyMin, MarketEmbDay, MarketEmbWeek
)
from xyz.finazon_service.market_analysis import (
    classify_trend_strength, classify_volatility_regime, classify_momentum_phase,
    identify_technical_signals, assess_risk_level
)
from xyz.llm.embedding_generator import get_embedding, num_tokens


# --------- Market Calendar & Filtering ---------

def is_market_open_day(dt):
    """
    Basic market open check - excludes weekends.
    Can be enhanced with holiday calendar if needed.
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return dt.weekday() < 5  # Monday=0, Friday=4


def is_trading_session(dt, volume=None):
    """
    Enhanced check for actual trading activity.
    Returns True if:
    1. Market day (not weekend)
    2. Has volume > 0 (optional check)
    """
    if not is_market_open_day(dt):
        return False

    if volume is not None and volume <= 0:
        return False

    return True


def get_market_open_periods(session, ticker_id, freq, recent_periods=50):
    """
    Returns DataFrame of periods with actual trading data only.
    Filters out weekends, holidays, and zero-volume periods.
    """
    # Query more data than needed to account for filtering
    oversample_factor = 1.5 if freq in ['30T', 'H'] else 2.0
    query_limit = int(recent_periods * oversample_factor)

    hist_data = (
        session.query(HistoricalData)
        .filter_by(ticker_id=ticker_id)
        .filter(HistoricalData.volume > 0)  # Only periods with trading
        .order_by(HistoricalData.timestamp.desc())
        .limit(query_limit)
        .all()
    )

    if not hist_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': hd.timestamp,
        'datetime': pd.to_datetime(hd.timestamp, unit='s'),
        'open': hd.open,
        'close': hd.close,
        'high': hd.high,
        'low': hd.low,
        'volume': hd.volume
    } for hd in hist_data])

    # Filter for market open days only
    df = df[df['datetime'].apply(is_market_open_day)]

    # Additional volume filter (redundant but safe)
    df = df[df['volume'] > 0]

    if df.empty:
        return df

    # Sort chronologically for proper aggregation
    df = df.sort_values('datetime')

    # Take only the most recent periods after filtering
    df = df.tail(recent_periods)

    logger.info(f"Filtered to {len(df)} market-open periods from {len(hist_data)} total records")
    return df


# --------- Helper Functions ---------

def to_yyyy_mm_dd(dt):
    """Converts datetime/date to 'YYYY-MM-DD' format."""
    if isinstance(dt, datetime) or isinstance(dt, date):
        return dt.strftime('%Y-%m-%d')
    elif isinstance(dt, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(dt, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        raise ValueError(f"Unknown date format: {dt}")
    else:
        raise TypeError(f"Unsupported type for date conversion: {type(dt)}")


def aggregate_metrics_with_market_filter(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Enhanced aggregation that only processes market-open periods.
    """
    try:
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Filter for market open periods before aggregation
        df = df[df['datetime'].apply(is_market_open_day)]
        df = df[df['volume'] > 0]  # Only periods with actual trading

        if df.empty:
            logger.warning("No market-open periods found after filtering")
            return pd.DataFrame()

        df.set_index('datetime', inplace=True)
        df.index.name = 'datetime'

        # Define aggregation dictionary for available columns
        base_agg_dict = {
            'close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
            'open': lambda x: x.iloc[0] if len(x) > 0 else np.nan,
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
        }

        # Add computed metrics if available
        computed_metrics_agg = {
            # Moving averages
            'sma': 'mean', 'ema': 'mean', 'dema': 'mean', 'tema': 'mean',
            'wma': 'mean', 'trima': 'mean', 'kama': 'mean', 'mama': 'mean', 't3': 'mean',
            'sma_20': 'mean', 'ema_20': 'mean', 'dema_20': 'mean', 'tema_20': 'mean',
            'wma_20': 'mean', 'trima_20': 'mean',

            # Returns and volatility
            'log_return': 'mean', 'historical_volatility': 'mean',
            'realized_volatility': 'mean', 'hourly_return': 'mean',
            'typical_price': 'mean', 'price_change': 'mean', 'price_change_pct': 'mean',

            # Technical indicators
            'macd': 'mean', 'macd_signal': 'mean', 'macd_hist': 'mean',
            'rsi': 'mean', 'stoch': 'mean', 'stochrsi': 'mean', 'stoch_k': 'mean', 'stoch_d': 'mean',
            'willr': 'mean', 'adx': 'mean', 'adxr': 'mean', 'apo': 'mean', 'ppo': 'mean', 'mom': 'mean',
            'bop': 'mean', 'cci': 'mean', 'cmo': 'mean', 'roc': 'mean', 'rocr': 'mean', 'aroon': 'mean',
            'aroonosc': 'mean', 'mfi': 'mean', 'trix': 'mean', 'ultosc': 'mean',

            # Bollinger Bands
            'bollinger_upper': 'mean', 'bollinger_lower': 'mean', 'bollinger_width': 'mean',

            # Volume and other metrics
            'obv': 'sum', 'cmf': 'mean', 'z_score': 'mean', 'ewma_score': 'mean',
            'sharpe_ratio': 'mean', 'sortino_ratio': 'mean', 'max_drawdown': 'min',
            'var': 'mean', 'cvar': 'mean', 'fama': 'mean'
        }

        # Only include columns that exist in the DataFrame
        agg_dict = {k: v for k, v in {**base_agg_dict, **computed_metrics_agg}.items()
                    if k in df.columns}

        # Perform aggregation
        if freq.startswith('W'):
            agg_df = df.resample(freq, label='left', closed='left').agg(agg_dict)
        else:
            agg_df = df.resample(freq).agg(agg_dict)

        # Filter out periods without basic OHLC data
        agg_df = agg_df.dropna(subset=['open', 'close'])

        # Additional filter: only keep periods that had actual trading volume
        if 'volume' in agg_df.columns:
            agg_df = agg_df[agg_df['volume'] > 0]

        if isinstance(agg_df, pd.Series):
            agg_df = agg_df.to_frame()

        agg_df = agg_df.reset_index()

        logger.info(f"Aggregated {len(df)} raw periods into {len(agg_df)} {freq} periods")
        return agg_df

    except Exception as e:
        logger.error(f"Error during market-filtered aggregation: {e}")
        logger.error(f"DataFrame info - columns: {df.columns.tolist() if not df.empty else 'empty'}")
        traceback.print_exc()
        raise


def make_market_state(row):
    """Create market state from aggregated row data."""
    state = {
        'trend_strength': classify_trend_strength(row),
        'volatility_regime': classify_volatility_regime(row),
        'momentum_phase': classify_momentum_phase(row),
        'technical_signals': identify_technical_signals(row),
        'market_position': 'N/A',
        'risk_level': assess_risk_level(row),
        'market_summary': make_market_summary(row)
    }
    return state


def make_market_summary(row):
    """Generate concise market summary."""
    return (
        f"Trend: {classify_trend_strength(row)}, "
        f"Volatility: {classify_volatility_regime(row)}, "
        f"Momentum: {classify_momentum_phase(row)}, "
        f"Signals: {identify_technical_signals(row)}, "
        f"Risk: {assess_risk_level(row)}"
    )


def fetch_news_for_period(ticker_symbol, company_name, period_start, period_end):
    """Fetch and summarize news for a given period."""
    try:
        articles = get_ticker_news_polygon(ticker_symbol, limit=12,
                                           published_from=period_start, published_to=period_end)
        relevant_articles = filter_articles(articles, ticker_symbol, company_name, scrape_cache=scrape_cache)
        extracted_data = extract_article_data(relevant_articles)
        text = create_news_summary_for_period(extracted_data, company_name, 512)
        logger.debug(f"News summary for {ticker_symbol}: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error fetching news for {ticker_symbol}: {e}")
        return "No news available"


def should_process_frequency(freq, force_all=False):
    """
    Smart timing logic to determine if a frequency should be processed.
    Returns (should_process, reason)
    """
    if force_all:
        return True, "Force mode enabled"

    try:
        market_tz = pytz.timezone('America/New_York')
        now = datetime.now(market_tz)
        current_time = now.time()
        current_weekday = now.weekday()  # 0=Monday, 6=Sunday

        if freq == '30T':
            # 30-min data: process during market hours and shortly after
            if current_weekday < 5:  # Weekday
                market_hours = dt_time(9, 30) <= current_time <= dt_time(17, 0)
                after_hours = dt_time(17, 0) <= current_time <= dt_time(18, 0)
                if market_hours or after_hours:
                    return True, "Market hours or after-hours window"
            return False, "Outside 30-min processing window"

        elif freq == 'D':
            # Daily: process after market close, pre-market, or weekends
            if current_weekday >= 5:  # Weekend
                weekend_hours = [9, 15, 21]
                if current_time.hour in weekend_hours:
                    return True, "Weekend processing window"
            else:  # Weekday
                after_market = current_time >= dt_time(16, 30)
                pre_market = current_time <= dt_time(9, 0)
                lunch_break = dt_time(12, 0) <= current_time <= dt_time(13, 30)
                if after_market or pre_market or lunch_break:
                    return True, "Off-market hours"
            return False, "Outside daily processing window"

        elif freq == 'W':
            # Weekly: process on weekends or Monday morning
            if current_weekday >= 5:  # Weekend
                weekend_hours = [10, 16]
                if current_time.hour in weekend_hours:
                    return True, "Weekend weekly window"
            elif current_weekday == 0 and current_time <= dt_time(10, 0):  # Monday morning
                return True, "Monday morning weekly processing"
            elif current_weekday in [2, 4] and current_time >= dt_time(19, 0):  # Wed/Fri evening
                return True, "Weekday evening weekly processing"
            return False, "Outside weekly processing window"

    except Exception as e:
        logger.warning(f"Timing logic failed: {e}. Using conservative defaults.")
        return freq == '30T', "Conservative fallback"

    return False, "Default: skip processing"


# --------- Main Pipeline ---------

def process_aggregated_embeddings(ticker_symbol, company_name, force_all=False):
    """
    Enhanced aggregation pipeline with market-aware filtering.
    Only processes periods when markets were actually open and trading occurred.
    """
    with get_db_session() as session:
        ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
        if not ticker:
            logger.error(f"Ticker {ticker_symbol} not found.")
            return

        # Initialize timeline pin manager
        pin_manager = TimelinePinManager()

        # Determine which frequencies to process
        frequencies = [
            ('30T', MarketEmbThirtyMin, 50),
            ('D', MarketEmbDay, 30),
            ('W-MON', MarketEmbWeek, 12)
        ]

        processing_plan = []
        for freq, table, periods in frequencies:
            should_process, reason = should_process_frequency(freq, force_all)
            if should_process:
                processing_plan.append((freq, table, periods, reason))
            else:
                logger.info(f"⏭️ Skipping {freq} for {ticker_symbol}: {reason}")

        if not processing_plan:
            logger.info(f"🚫 No frequencies to process for {ticker_symbol}")
            return

        logger.info(f"📋 Processing plan for {ticker_symbol}: {[f[0] for f in processing_plan]}")

        # Fetch computed metrics with market filtering
        metrics_query = (
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
            .filter(HistoricalData.volume > 0)  # Only trading periods
            .order_by(HistoricalData.timestamp.desc())
            .limit(200)  # Get enough data for all frequencies
        )

        # Convert to DataFrame with market filtering
        raw_data = metrics_query.all()
        if not raw_data:
            logger.warning(f"No computed metrics with trading volume found for {ticker_symbol}")
            return

        # Create base DataFrame
        df = pd.DataFrame(raw_data, columns=[col['name'] for col in metrics_query.column_descriptions])

        # Apply market-open filter
        df = df[df['datetime'].apply(is_market_open_day)]
        df = df[df['volume'] > 0]

        if df.empty:
            logger.warning(f"No market-open periods found for {ticker_symbol}")
            return

        logger.info(f"📊 Processing {len(df)} market-open data points for {ticker_symbol}")

        # Store daily summaries for weekly aggregation
        daily_summaries = {}

        # Process each frequency
        for freq, Table, recent_periods, reason in processing_plan:
            logger.info(f"🔄 Processing {freq} aggregations for {ticker_symbol} ({reason})")

            # Get market-filtered aggregated data
            agg_df = aggregate_metrics_with_market_filter(df.copy(), freq)

            if agg_df.empty:
                logger.warning(f"No aggregated data for {freq} frequency")
                continue

            # Limit to recent periods
            agg_df = agg_df.tail(recent_periods)

            processed_count = 0
            skipped_count = 0

            for _, row in agg_df.iterrows():
                row_dict = row.to_dict()
                period_start = row['datetime']

                # Check if already exists
                exists = (
                    session.query(Table)
                    .filter_by(ticker_id=ticker.id, period_start=period_start)
                    .first()
                )
                if exists:
                    skipped_count += 1
                    continue

                # Generate market state
                state = make_market_state(row_dict)

                # Handle news and flags based on frequency
                flags = []
                priority_score = 0
                should_pin = False

                if freq == '30T':
                    period_end = period_start + timedelta(minutes=30)
                    news_blob = "30T"  # No news for 30min
                elif freq == 'D':
                    period_end = period_start + timedelta(days=1)
                    # Enhanced news with flags
                    flagged_news = fetch_news_for_period_with_flags(
                        ticker_symbol, company_name,
                        to_yyyy_mm_dd(period_start), to_yyyy_mm_dd(period_end)
                    )
                    news_blob, flags = parse_news_flags(flagged_news)
                    priority_score = pin_manager.calculate_priority_score(flags)
                    should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

                    # Store for weekly use
                    if news_blob and news_blob.strip() and news_blob.strip().lower() != "false":
                        daily_summaries[to_yyyy_mm_dd(period_start)] = news_blob.strip()

                elif freq == 'W-MON':
                    period_end = period_start + timedelta(weeks=1)
                    week_start = period_start.date()
                    week_end = period_end.date()

                    # Weekly news with cached daily summaries
                    flagged_weekly = fetch_and_summarize_weekly_articles_cached_with_flags(
                        ticker_symbol, company_name, week_start, week_end, daily_summaries
                    )
                    news_blob, flags = parse_news_flags(flagged_weekly)
                    priority_score = pin_manager.calculate_priority_score(flags)
                    should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

                # Create embedding with token management
                text_to_embed = f"{state['market_summary']}\nNews: {news_blob}"

                # Token truncation logic
                market_tokens = num_tokens(state["market_summary"])
                news_prefix_tokens = num_tokens("\nNews: ")
                news_tokens = num_tokens(news_blob)
                total_tokens = market_tokens + news_prefix_tokens + news_tokens

                if total_tokens > 8192:
                    available_tokens = 8192 - market_tokens - news_prefix_tokens
                    encoding = tiktoken.encoding_for_model(OAI.gpt4o)
                    news_tokens_list = encoding.encode(news_blob)
                    truncated_news = encoding.decode(news_tokens_list[:available_tokens])
                    text_to_embed = f"{state['market_summary']}\nNews: {truncated_news}"

                embedding_vector = get_embedding(text_to_embed)

                # Create record
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
                    news_flags=','.join(flags) if flags else None,
                    priority_score=priority_score,
                    is_pinned=should_pin,
                    embedding_vector=embedding_vector
                )
                session.add(emb)
                processed_count += 1

                if flags:
                    logger.info(f"🏷️ {freq} flags for {period_start}: {flags}, priority: {priority_score}")

            logger.info(f"✅ {freq}: processed {processed_count} new, skipped {skipped_count} existing")

        # Commit all changes
        try:
            session.commit()
            logger.info(f"🎯 Aggregation completed successfully for {ticker_symbol}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error committing aggregation data for {ticker_symbol}: {e}")
            raise


# ---------- Example Usage ----------
if __name__ == "__main__":
    ticker_symbol = "AAPL"
    company_name = "Apple Inc."

    # Test with force_all=False to use smart timing
    process_aggregated_embeddings(ticker_symbol, company_name, force_all=False)

    # Or force all frequencies
    # process_aggregated_embeddings(ticker_symbol, company_name, force_all=True)