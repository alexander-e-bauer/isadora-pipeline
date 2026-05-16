import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import logger
from xyz.finazon_service.sql_service import (
    get_db_session, Ticker, HistoricalData, ComputedMetrics
)
from xyz.finazon_service.metrics import compute_batch_metrics


def backfill_recent_metrics(ticker_symbol, lookback_periods=200, update_periods=50):
    """
    Recalculate metrics for the most recent periods to ensure consistency.
    Uses vectorized batch processing for efficiency.
    """
    session = get_db_session()
    try:
        ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
        if not ticker:
            logger.warning(f"Ticker {ticker_symbol} not found")
            return

        # Get recent historical data
        recent_hist = (
            session.query(HistoricalData)
            .filter_by(ticker_id=ticker.id)
            .order_by(HistoricalData.timestamp.desc())
            .limit(lookback_periods)
            .all()
        )
        if not recent_hist:
            logger.warning(f"No historical data for {ticker_symbol}")
            return

        # Compute metrics for each row
        for hist in recent_hist[:update_periods]:
            metrics = compute_batch_metrics(hist)
            # Upsert metrics
            cm = (
                session.query(ComputedMetrics)
                .filter_by(historical_data_id=hist.id)
                .first()
            )
            if cm:
                for k, v in metrics.items():
                    setattr(cm, k, v)
            else:
                cm = ComputedMetrics(historical_data_id=hist.id, **metrics)
                session.add(cm)
        session.commit()
        logger.info(f"Backfill complete for {ticker_symbol}")
    except Exception as e:
        logger.error(f"Backfill error for {ticker_symbol}: {e}")
        session.rollback()
    finally:
        session.close()


def backfill_all_tickers(max_tickers=None, lookback_periods=200, update_periods=50):
    """
    Backfill metrics for all tickers in the database.

    Args:
        max_tickers: Maximum number of tickers to process (None for all)
        lookback_periods: Number of periods to look back for context
        update_periods: Number of most recent periods to update
    """
    try:
        with get_db_session() as session:
            query = session.query(Ticker.symbol)
            if max_tickers:
                query = query.limit(max_tickers)

            tickers = [row[0] for row in query.all()]

            logger.info(f"Starting backfill for {len(tickers)} tickers")

            success_count = 0
            for i, ticker_symbol in enumerate(tickers, 1):
                logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker_symbol}")

                if backfill_recent_metrics(ticker_symbol, lookback_periods, update_periods):
                    success_count += 1

                # Progress update every 10 tickers
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(tickers)} tickers processed, {success_count} successful")

            logger.info(f"Backfill completed: {success_count}/{len(tickers)} tickers successful")
            return success_count

    except Exception as e:
        logger.error(f"Error in backfill_all_tickers: {str(e)}")
        return 0


def backfill_missing_metrics(ticker_symbol=None, batch_size=1000):
    """
    Find and compute metrics for historical data that doesn't have computed metrics.

    Args:
        ticker_symbol: Specific ticker to process (None for all)
        batch_size: Number of records to process in each batch
    """
    try:
        with get_db_session() as session:
            if ticker_symbol:
                # Process specific ticker
                from xyz.finazon_service.metrics import compute_metrics_for_ticker
                compute_metrics_for_ticker(session, ticker_symbol, batch_size)
            else:
                # Process all tickers with missing metrics
                query = """
                        SELECT DISTINCT t.symbol
                        FROM tickers t
                                 JOIN historical_data hd ON t.id = hd.ticker_id
                                 LEFT JOIN computed_metrics cm ON hd.id = cm.historical_data_id
                        WHERE cm.id IS NULL
                        ORDER BY t.symbol \
                        """

                result = session.execute(query)
                tickers_with_missing = [row[0] for row in result.fetchall()]

                logger.info(f"Found {len(tickers_with_missing)} tickers with missing metrics")

                for ticker in tickers_with_missing:
                    logger.info(f"Processing missing metrics for {ticker}")
                    from xyz.finazon_service.metrics import compute_metrics_for_ticker
                    compute_metrics_for_ticker(session, ticker, batch_size)

    except Exception as e:
        logger.error(f"Error in backfill_missing_metrics: {str(e)}")
        raise


def validate_metrics_consistency(ticker_symbol, sample_size=100):
    """
    Validate that computed metrics are consistent with historical data.

    Args:
        ticker_symbol: Ticker to validate
        sample_size: Number of recent records to validate
    """
    try:
        with get_db_session() as session:
            ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
            if not ticker:
                logger.error(f"Ticker {ticker_symbol} not found")
                return False

            # Get recent data with metrics
            query = """
                    SELECT hd.*, cm.*
                    FROM historical_data hd
                             JOIN computed_metrics cm ON hd.id = cm.historical_data_id
                    WHERE hd.ticker_id = :ticker_id
                    ORDER BY hd.timestamp DESC LIMIT :limit \
                    """

            df = pd.read_sql(query, session.bind, params={
                'ticker_id': ticker.id,
                'limit': sample_size
            })

            if df.empty:
                logger.warning(f"No data with metrics found for {ticker_symbol}")
                return False

            # Sort chronologically for validation
            df = df.sort_values('timestamp')

            # Validate some basic calculations
            calculated_returns = np.log(df['close'] / df['close'].shift(1))
            stored_returns = df['log_return']

            # Compare (allowing for small floating point differences)
            diff = np.abs(calculated_returns - stored_returns).dropna()
            max_diff = diff.max()

            if max_diff > 1e-10:  # Very small tolerance
                logger.warning(f"Metrics inconsistency detected for {ticker_symbol}, max diff: {max_diff}")
                return False
            else:
                logger.info(f"Metrics validation passed for {ticker_symbol}")
                return True

    except Exception as e:
        logger.error(f"Error validating metrics for {ticker_symbol}: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Backfill metrics for tickers')
    parser.add_argument('--ticker', type=str, help='Specific ticker to backfill')
    parser.add_argument('--all', action='store_true', help='Backfill all tickers')
    parser.add_argument('--missing', action='store_true', help='Backfill only missing metrics')
    parser.add_argument('--validate', type=str, help='Validate metrics for specific ticker')
    parser.add_argument('--lookback', type=int, default=200, help='Lookback periods')
    parser.add_argument('--update', type=int, default=50, help='Update periods')

    args = parser.parse_args()

    if args.ticker:
        backfill_recent_metrics(args.ticker, args.lookback, args.update)
    elif args.all:
        backfill_all_tickers(lookback_periods=args.lookback, update_periods=args.update)
    elif args.missing:
        backfill_missing_metrics()
    elif args.validate:
        validate_metrics_consistency(args.validate)
    else:
        print("Please specify --ticker, --all, --missing, or --validate")