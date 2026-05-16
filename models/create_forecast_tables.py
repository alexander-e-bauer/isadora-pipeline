# create_forecast_tables.py
from xyz.finazon_service.sql_service import engine, Base, ForecastMetrics, ModelPerformance

def create_forecast_tables():
    """Create forecast-related tables"""
    try:
        # Create only the new tables
        ForecastMetrics.__table__.create(engine, checkfirst=True)
        ModelPerformance.__table__.create(engine, checkfirst=True)
        print("Forecast tables created successfully!")
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_forecast_tables()
