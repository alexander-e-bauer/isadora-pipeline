"""xyz.tenant — read-only tenant schema mirror for the engine.

Engine reads from the same Postgres as server (firms, users, clients,
accounts, strategies, deployments).  Engine WRITES only to the events
table via emit_event — all other tenant tables are read-only from
engine's perspective.  Server's Alembic migrations own DDL.
"""
