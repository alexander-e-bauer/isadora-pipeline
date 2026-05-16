import pandas as pd
import numpy as np
import traceback
import tiktoken
from datetime import datetime, date, timedelta
from config import logger, OAI
from xyz.finazon_service.api_service import (
    create_weekly_summary_for_period, extract_article_data,
    create_news_summary_for_period, filter_articles, scrape_cache,
    get_ticker_news_polygon,
    # Add the new enhanced functions:
    fetch_news_for_period_with_flags,
    fetch_and_summarize_weekly_articles_cached_with_flags,
    TimelinePinManager, parse_news_flags
)
from xyz.finazon_service.sql_service import (
  get_db_session, Ticker, HistoricalData, ComputedMetrics,  # <- Use get_db_session instead of Session
  MarketEmbThirtyMin, MarketEmbDay, MarketEmbWeek
)
from xyz.finazon_service.market_analysis import (
  classify_trend_strength, classify_volatility_regime, classify_momentum_phase,
  identify_technical_signals, assess_risk_level
)
from xyz.llm.embedding_generator import get_embedding, num_tokens

# --------- Helper: Aggregate Computed Metrics ---------

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


def aggregate_metrics_dep(df: pd.DataFrame, freq: str) -> pd.DataFrame:
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
      if freq.startswith('W'):
          agg_df = df.resample(freq, label='left', closed='left').agg(agg_dict_filtered)
      else:
          agg_df = df.resample(freq).agg(agg_dict_filtered)

      agg_df = agg_df.dropna(subset=['open', 'close'])

      if isinstance(agg_df, pd.Series):
          agg_df = agg_df.to_frame()
      agg_df = agg_df.reset_index()
      return agg_df

  except Exception as e:
      logger.error(f"Error during aggregation: {e}")
      logger.error(f"df.columns: {df.columns}")
      logger.error(f"df.index.name: {df.index.name}")
      traceback.print_exc()
      raise


def aggregate_metrics(session, ticker_symbol, freq='D', recent_periods=50):
    """
    Aggregate metrics for recent periods and upsert into embedding tables.
    """
    ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
    if not ticker:
        logger.warning(f"Ticker {ticker_symbol} not found for aggregation")
        return

    # Get recent metrics
    hist_data = (
        session.query(HistoricalData)
        .filter_by(ticker_id=ticker.id)
        .order_by(HistoricalData.timestamp.desc())
        .limit(recent_periods)
        .all()
    )
    if not hist_data:
        logger.warning(f"No data for {ticker_symbol} during aggregation")
        return

    df = pd.DataFrame([hd.__dict__ for hd in hist_data])
    agg_df = aggregate_metrics(df, freq)

    # Upsert into embedding tables
    Model = {
        'D': MarketEmbDay,
        'W': MarketEmbWeek,
        '30min': MarketEmbThirtyMin,
    }[freq]

    for _, row in agg_df.iterrows():
        emb_row = (
            session.query(Model)
            .filter_by(ticker_id=ticker.id, period_start=row['period_start'])
            .first()
        )
        if emb_row:
            # Update fields
            for k in row.keys():
                setattr(emb_row, k, row[k])
        else:
            emb_row = Model(ticker_id=ticker.id, **row)
            session.add(emb_row)
    session.commit()
    logger.info(f"Aggregation complete for {ticker_symbol}, freq={freq}")


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
def fetch_news_for_period(ticker_symbol, company_name, period_start, period_end):
  try:
      articles = get_ticker_news_polygon(ticker_symbol, limit=12, published_from=period_start, published_to=period_end)
      relevant_articles = filter_articles(articles, ticker_symbol, company_name, scrape_cache=scrape_cache)
      extracted_data = extract_article_data(relevant_articles)
      text = create_news_summary_for_period(extracted_data, company_name, 512)
      logger.debug(f"fetch_news_for_period summary output: {text}")
      return text
  except Exception as e:
      logger.error(f"Error fetching news for {ticker_symbol}: {e}")
      return "No news available"

# --------- Main Pipeline ---------


def process_aggregated_embeddings(ticker_symbol, company_name, force_all=True):
    """
    Aggregates computed metrics with smart timing to avoid unnecessary processing.
    Daily and weekly embeddings are processed only once per period.
    Enhanced with timeline flag processing.
    """
    from datetime import datetime, time as dt_time
    import pytz

    with get_db_session() as session:
        ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
        if not ticker:
            logger.error(f"Ticker {ticker_symbol} not found.")
            return

        # Initialize timeline pin manager
        pin_manager = TimelinePinManager()

        # Smart timing logic with daily and weekly optimization
        try:
            market_tz = pytz.timezone('America/New_York')
            now = datetime.now(market_tz)
            current_time = now.time()
            current_weekday = now.weekday()  # 0=Monday, 6=Sunday

            # Always process 30min data (this is real-time)
            process_30min = True

            if force_all:
                # Manual run or forced mode - aggregate everything
                process_daily = True
                process_weekly = True
                logger.info(f"🔄 Force mode: aggregating all timeframes for {ticker_symbol}")
            else:
                # === DAILY PROCESSING LOGIC ===
                process_daily = False

                # Check if we already processed daily data today
                today_start = datetime.combine(now.date(), dt_time(0, 0))
                today_start = market_tz.localize(today_start)

                recent_daily = (
                    session.query(MarketEmbDay)
                    .filter(
                        MarketEmbDay.ticker_id == ticker.id,
                        MarketEmbDay.created_at >= today_start
                    )
                    .first()
                )

                if recent_daily:
                    process_daily = False
                    logger.info(f"📅 Daily data already processed today for {ticker_symbol}")
                else:
                    # Process daily only during specific hours to avoid spam
                    # Good times: after market close, pre-market, lunch break, weekends

                    if current_weekday >= 5:  # Weekend
                        # Process daily on weekends (catching up on Friday's data, etc.)
                        daily_weekend_hours = [9, 15, 21]  # 9 AM, 3 PM, 9 PM
                        if current_time.hour in daily_weekend_hours:
                            process_daily = True
                            logger.info(f"📅 Weekend daily processing window for {ticker_symbol}")
                        else:
                            logger.info(f"📅 Outside weekend daily window for {ticker_symbol}")
                    else:  # Weekday
                        # Process daily during off-market hours or lunch
                        after_market = current_time >= dt_time(16, 30)  # After 4:30 PM
                        pre_market = current_time <= dt_time(9, 0)  # Before 9:00 AM
                        lunch_break = dt_time(12, 0) <= current_time <= dt_time(13, 30)  # Lunch 12-1:30 PM

                        if after_market or pre_market or lunch_break:
                            process_daily = True
                            logger.info(f"📅 Off-market hours: processing daily for {ticker_symbol}")
                        else:
                            logger.info(f"📅 Market hours: skipping daily for {ticker_symbol}")

                # === WEEKLY PROCESSING LOGIC ===
                process_weekly = False

                # Check if we already processed weekly data this week
                # Get start of current week (Monday)
                days_since_monday = current_weekday
                week_start = now.date() - timedelta(days=days_since_monday)
                week_start_dt = datetime.combine(week_start, dt_time(0, 0))
                week_start_dt = market_tz.localize(week_start_dt)

                recent_weekly = (
                    session.query(MarketEmbWeek)
                    .filter(
                        MarketEmbWeek.ticker_id == ticker.id,
                        MarketEmbWeek.created_at >= week_start_dt
                    )
                    .first()
                )

                if recent_weekly:
                    process_weekly = False
                    logger.info(f"📊 Weekly data already processed this week for {ticker_symbol}")
                else:
                    # Process weekly only during specific times
                    if current_weekday >= 5:  # Weekend
                        # Weekend processing windows
                        weekend_weekly_hours = [10, 16]  # 10 AM, 4 PM
                        if current_time.hour in weekend_weekly_hours:
                            process_weekly = True
                            logger.info(f"📊 Weekend weekly processing window for {ticker_symbol}")
                        else:
                            logger.info(f"📊 Outside weekend weekly window for {ticker_symbol}")

                    elif current_weekday == 0:  # Monday
                        # Monday morning - good time to process last week's data
                        if current_time <= dt_time(10, 0):
                            process_weekly = True
                            logger.info(f"📊 Monday morning: processing weekly for {ticker_symbol}")
                        else:
                            logger.info(f"📊 Monday late: skipping weekly for {ticker_symbol}")
                    else:
                        # Weekday evening processing (very selective)
                        if (current_weekday in [2, 4] and  # Wednesday, Friday
                                current_time >= dt_time(19, 0)):  # After 7 PM
                            process_weekly = True
                            logger.info(f"📊 Weekday evening: processing weekly for {ticker_symbol}")
                        else:
                            logger.info(f"📊 Weekday: skipping weekly for {ticker_symbol}")

        except Exception as e:
            # If timezone logic fails, default to conservative processing
            logger.warning(f"⚠️ Timezone logic failed for {ticker_symbol}: {e}. Using conservative defaults.")
            process_30min = True
            process_daily = False  # Conservative: don't spam daily
            process_weekly = False  # Conservative: don't spam weekly

        logger.info(
            f"🎯 Aggregation plan for {ticker_symbol}: "
            f"30min={process_30min}, daily={process_daily}, weekly={process_weekly}"
        )

        # === PROCESSING SECTION ===
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
            logger.warning(f"No computed metrics found for {ticker_symbol}.")
            return

        # 2. Process based on timing decisions
        daily_summaries = {}
        frequencies_to_process = []

        if process_30min:
            frequencies_to_process.append(('30T', MarketEmbThirtyMin))

        if process_daily:
            frequencies_to_process.append(('d', MarketEmbDay))

        # Process 30min and daily together
        if frequencies_to_process:
            logger.info(f"🔄 Processing {len(frequencies_to_process)} frequency type(s) for {ticker_symbol}")

            for freq, Table in frequencies_to_process:
                logger.info(f"📊 Processing {freq} aggregations for {ticker_symbol}")

                agg_df = aggregate_metrics(df, freq)
                processed_count = 0
                skipped_count = 0

                for i, row in agg_df.iterrows():
                    row_dict = row.to_dict()
                    state = make_market_state(row_dict)
                    period_start = row['datetime']

                    # Check if this specific period already exists
                    exists = (
                        session.query(Table)
                        .filter_by(ticker_id=ticker.id, period_start=period_start)
                        .first()
                    )
                    if exists:
                        logger.debug(f"Skipping existing {freq} data for {period_start}")
                        skipped_count += 1
                        continue

                    # Initialize flag-related variables
                    flags = []
                    priority_score = 0
                    should_pin = False

                    if freq == '30T':
                        period_end = period_start + timedelta(hours=1)
                        news_blob = "30T"  # No news for 30min intervals
                    elif freq == 'd':
                        period_end = period_start + timedelta(days=1)

                        # Enhanced news processing with flags
                        flagged_news = fetch_news_for_period_with_flags(
                            ticker_symbol, company_name,
                            to_yyyy_mm_dd(period_start), to_yyyy_mm_dd(period_end)
                        )

                        # Parse flags and calculate priority
                        news_blob, flags = parse_news_flags(flagged_news)
                        priority_score = pin_manager.calculate_priority_score(flags)
                        should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

                        # Store daily summary for potential weekly use
                        if news_blob and news_blob.strip() and news_blob.strip().lower() != "false":
                            daily_summaries[to_yyyy_mm_dd(period_start)] = news_blob.strip()

                    # Create embedding and store
                    text_to_embed = f"{state['market_summary']}\nNews: {news_blob}"

                    # Token management (your existing logic)
                    market_summary_tokens = num_tokens(state["market_summary"])
                    news_prefix_tokens = num_tokens("\nNews: ")
                    news_blob_tokens = num_tokens(news_blob)
                    total_tokens = market_summary_tokens + news_prefix_tokens + news_blob_tokens

                    if total_tokens <= 8192:
                        text_to_embed = f"{state['market_summary']}\nNews: {news_blob}"
                    else:
                        tokens_available_for_news = 8192 - market_summary_tokens - news_prefix_tokens
                        encoding = tiktoken.encoding_for_model(OAI.gpt4o)
                        news_blob_tokens_list = encoding.encode(news_blob)
                        truncated_news_blob = encoding.decode(news_blob_tokens_list[:tokens_available_for_news])
                        text_to_embed = f"{state['market_summary']}\nNews: {truncated_news_blob}"

                    embedding_vector = get_embedding(text_to_embed)

                    # Create record with enhanced flag fields
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

                    # Log flag information for daily records
                    if freq == 'd' and flags:
                        logger.info(
                            f"📌 Daily flags for {period_start.date()}: {flags}, priority: {priority_score}, pinned: {should_pin}")

                logger.info(f"✅ {freq}: processed {processed_count} new, skipped {skipped_count} existing")

        # Process weekly separately
        if process_weekly:
            logger.info(f"📅 Processing weekly aggregations for {ticker_symbol}")

            agg_df = aggregate_metrics(df, 'W-MON')
            weekly_processed_count = 0
            weekly_skipped_count = 0

            for i, row in agg_df.iterrows():
                row_dict = row.to_dict()
                state = make_market_state(row_dict)
                period_start = row['datetime']
                period_end = period_start + timedelta(weeks=1)
                week_start = period_start.date()
                week_end = period_end.date()

                # Check if this specific weekly period already exists
                exists = (
                    session.query(MarketEmbWeek)
                    .filter_by(ticker_id=ticker.id, period_start=period_start)
                    .first()
                )
                if exists:
                    logger.debug(f"Skipping existing weekly data for {period_start}")
                    weekly_skipped_count += 1
                    continue

                # Get weekly news summary (using existing cached daily summaries)
                flagged_weekly = fetch_and_summarize_weekly_articles_cached_with_flags(
                    ticker_symbol, company_name, week_start, week_end, daily_summaries
                )
                news_blob, flags = parse_news_flags(flagged_weekly)

                # Calculate priority and pin status for weekly
                priority_score = pin_manager.calculate_priority_score(flags)
                should_pin = pin_manager.should_create_timeline_pin(flags, priority_score)

                # Create embedding and store
                text_to_embed = f"{state['market_summary']}\nNews: {news_blob}"

                # Token management (same as above)
                market_summary_tokens = num_tokens(state["market_summary"])
                news_prefix_tokens = num_tokens("\nNews: ")
                news_blob_tokens = num_tokens(news_blob)
                total_tokens = market_summary_tokens + news_prefix_tokens + news_blob_tokens

                if total_tokens <= 8192:
                    text_to_embed = f"{state['market_summary']}\nNews: {news_blob}"
                else:
                    tokens_available_for_news = 8192 - market_summary_tokens - news_prefix_tokens
                    encoding = tiktoken.encoding_for_model("gpt-4")
                    news_blob_tokens_list = encoding.encode(news_blob)
                    truncated_news_blob = encoding.decode(news_blob_tokens_list[:tokens_available_for_news])
                    text_to_embed = f"{state['market_summary']}\nNews: {truncated_news_blob}"

                embedding_vector = get_embedding(text_to_embed)

                # Create weekly record with enhanced flag fields
                emb = MarketEmbWeek(
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
                weekly_processed_count += 1

                # Log flag information for weekly records
                if flags:
                    logger.info(
                        f"📌 Weekly flags for {week_start}: {flags}, priority: {priority_score}, pinned: {should_pin}")

            logger.info(f"✅ Weekly: processed {weekly_processed_count} new, skipped {weekly_skipped_count} existing")

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
  from tqdm import tqdm
  ticker_symbol = "AAPL"
  company_name = "Apple"
  from_date = "2025-06-01"
  to_date = "2025-06-07"

  # Convert string dates to datetime.date objects
  start_date = datetime.strptime(from_date, "%Y-%m-%d").date()
  end_date = datetime.strptime(to_date, "%Y-%m-%d").date()

  # Step 1: Build daily summaries dict
  daily_summaries = {}
  num_days = (end_date - start_date).days
  for i in range(num_days+1):
      current_day = start_date + timedelta(days=i)
      day_str = current_day.strftime("%Y-%m-%d")
      news = fetch_news_for_period(ticker_symbol, company_name, day_str, day_str)
      # Don't include "False" or empty
      if news and news.strip() and news.strip().lower() != "false":
          daily_summaries[day_str] = news.strip()

  # Step 2: Use the daily_summaries dict for weekly summary
  weekly_news = fetch_and_summarize_weekly_articles_cached_with_flags(
      ticker_symbol, company_name, start_date, end_date, daily_summaries
  )

  print(f"Final weekly summary:\n{weekly_news}")