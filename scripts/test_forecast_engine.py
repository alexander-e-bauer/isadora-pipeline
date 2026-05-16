# scripts/test_forecast_engine.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xyz.finazon_service.forecast_engine import ForecastEngine
from config import logger


def test_forecast():
    try:
        logger.info("Starting forecast engine test...")
        engine = ForecastEngine()

        # Test with a ticker you have data for - replace 'AAPL' with a ticker you know exists in your DB
        test_symbol = 'AAPL'  # Change this to a ticker you have in your database
        logger.info(f"Testing forecast generation for {test_symbol}")

        success = engine.generate_forecasts_for_ticker(test_symbol, horizons=[1, 7])

        if success:
            logger.info("✅ Forecast generation successful!")
            print("✅ Forecast generation successful!")
        else:
            logger.error("❌ Forecast generation failed!")
            print("❌ Forecast generation failed!")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_forecast()
