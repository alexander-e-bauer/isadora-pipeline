"""Shared SQLAlchemy declarative Base for the engine's finazon data tables.

Extracted into its own module so that downstream packages (e.g.
xyz.polygon_service.models) can import the Base without triggering
psycopg2 / the live Postgres engine creation that lives in sql_service.py.
"""
from sqlalchemy.orm import declarative_base

Base = declarative_base()
