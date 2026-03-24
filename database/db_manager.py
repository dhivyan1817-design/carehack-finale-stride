"""
STRIDE — SQLite Database Manager

Handles all database operations for STRIDE.
Two tables:
  - sensor_readings : raw sensor data as received (one row per reading)
  - risk_results    : ML outputs (SRI, DPR, ETC, urgency) per zone per timestamp

Design principle:
  - This layer sits between ingestion and feature engineering.
  - Nothing above this layer (features, model, decision, dashboard) knows or
    cares about SQL. They only ever see pandas DataFrames.
  - To switch to PostgreSQL later: change _connect() only. Nothing else changes.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "stride.db")


def _connect():
    """Returns a SQLite connection. Swap this for PostgreSQL later if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates tables if they don't already exist.
    Safe to call on every startup — will not overwrite existing data.
    """
    conn = _connect()
    c = conn.cursor()

    # Raw sensor readings table
    c.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at     TEXT    NOT NULL,
            zone_id         TEXT    NOT NULL,
            deformation_mm  REAL,
            gap_mm          REAL,
            vibration_level REAL,
            temperature     REAL,
            load_estimate   REAL,
            data_source     TEXT    DEFAULT 'simulated'
        )
    """)

    # ML risk results table
    c.execute("""
        CREATE TABLE IF NOT EXISTS risk_results (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at         TEXT    NOT NULL,
            zone_id             TEXT    NOT NULL,
            sri                 REAL,
            dpr                 REAL,
            etc                 REAL,
            urgency_category    TEXT,
            urgency_explanation TEXT,
            deformation_mm      REAL,
            gap_mm              REAL,
            data_source         TEXT    DEFAULT 'simulated'
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialised at {DB_PATH}")


# ── Write operations ──────────────────────────────────────────────────────────

def insert_readings(df, data_source="simulated"):
    """
    Inserts a DataFrame of raw sensor readings into sensor_readings table.
    Expects columns: zone_id, timestamp, deformation_mm, gap_mm,
                     vibration_level, temperature, load_estimate
    """
    if df is None or df.empty:
        return 0

    conn = _connect()
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT INTO sensor_readings
                    (recorded_at, zone_id, deformation_mm, gap_mm,
                     vibration_level, temperature, load_estimate, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get('timestamp', datetime.now())),
                str(row['zone_id']),
                float(row.get('deformation_mm', 0)),
                float(row.get('gap_mm', 0)),
                float(row.get('vibration_level', 0)),
                float(row.get('temperature', 25.0)),
                float(row.get('load_estimate', 500.0)),
                data_source
            ))
            inserted += 1
        except Exception as e:
            print(f"[DB] Insert reading error: {e}")
            continue

    conn.commit()
    conn.close()
    return inserted


def insert_risk_results(df, data_source="simulated"):
    """
    Inserts ML risk results (latest per zone) into risk_results table.
    Expects columns: zone_id, timestamp, sri, dpr, etc,
                     urgency_category, urgency_explanation,
                     deformation_mm, gap_mm
    """
    if df is None or df.empty:
        return 0

    # Only store latest reading per zone to avoid redundant history
    latest = df

    conn = _connect()
    inserted = 0
    for _, row in latest.iterrows():
        try:
            etc_val = row.get('etc', None)
            if etc_val == float('inf') or etc_val > 9000:
                etc_val = None  # Store NULL for stable zones

            conn.execute("""
                INSERT INTO risk_results
                    (recorded_at, zone_id, sri, dpr, etc,
                     urgency_category, urgency_explanation,
                     deformation_mm, gap_mm, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get('timestamp', datetime.now())),
                str(row['zone_id']),
                float(row.get('sri', 0)),
                float(row.get('dpr', 0)),
                etc_val,
                str(row.get('urgency_category', 'Monitor')),
                str(row.get('urgency_explanation', '')),
                float(row.get('deformation_mm', 0)),
                float(row.get('gap_mm', 0)),
                data_source
            ))
            inserted += 1
        except Exception as e:
            print(f"[DB] Insert result error: {e}")
            continue

    conn.commit()
    conn.close()
    return inserted


# ── Read operations ───────────────────────────────────────────────────────────

def get_latest_risk_per_zone():
    """
    Returns the most recent risk result for each zone.
    Used by dashboard for the live overview panel.
    """
    conn = _connect()
    df = pd.read_sql_query("""
        SELECT r.*
        FROM risk_results r
        INNER JOIN (
            SELECT zone_id, MAX(recorded_at) AS max_time
            FROM risk_results
            GROUP BY zone_id
        ) latest ON r.zone_id = latest.zone_id AND r.recorded_at = latest.max_time
        ORDER BY sri DESC
    """, conn)
    conn.close()
    return df


def get_risk_history(zone_id=None, limit=2000):
    """
    Returns historical risk results for trend charts.
    Deduplicates by zone + timestamp so repeated main.py runs don't bloat chart.
    """
    conn = _connect()
    if zone_id:
        df = pd.read_sql_query("""
            SELECT zone_id, recorded_at, AVG(sri) as sri, AVG(dpr) as dpr,
                   urgency_category, data_source
            FROM risk_results
            WHERE zone_id = ?
            GROUP BY zone_id, recorded_at
            ORDER BY recorded_at ASC
            LIMIT ?
        """, conn, params=(zone_id, limit))
    else:
        df = pd.read_sql_query("""
            SELECT zone_id, recorded_at, AVG(sri) as sri, AVG(dpr) as dpr,
                   urgency_category, data_source
            FROM risk_results
            GROUP BY zone_id, recorded_at
            ORDER BY recorded_at ASC
            LIMIT ?
        """, conn, params=(limit,))
    conn.close()

    if not df.empty:
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
    return df


def get_sensor_history(zone_id=None, limit=500):
    """
    Returns raw sensor readings history.
    """
    conn = _connect()
    if zone_id:
        df = pd.read_sql_query("""
            SELECT * FROM sensor_readings
            WHERE zone_id = ?
            ORDER BY recorded_at ASC
            LIMIT ?
        """, conn, params=(zone_id, limit))
    else:
        df = pd.read_sql_query("""
            SELECT * FROM sensor_readings
            ORDER BY recorded_at ASC
            LIMIT ?
        """, conn, params=(limit,))
    conn.close()

    if not df.empty:
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
    return df


def get_alert_history(limit=100):
    """
    Returns only zones that triggered Immediate Attention or Repair Soon.
    Useful for the judge demo — shows the system caught real events.
    """
    conn = _connect()
    df = pd.read_sql_query("""
        SELECT recorded_at, zone_id, sri, dpr, etc,
               urgency_category, urgency_explanation, data_source
        FROM risk_results
        WHERE urgency_category IN ('Immediate Attention', 'Repair Soon')
        ORDER BY recorded_at DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()

    if not df.empty:
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
    return df


def get_db_stats():
    """
    Returns basic stats about the database for the dashboard info panel.
    """
    conn = _connect()
    c = conn.cursor()

    stats = {}
    c.execute("SELECT COUNT(*) FROM sensor_readings")
    stats['total_readings'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM risk_results")
    stats['total_risk_records'] = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT zone_id) FROM risk_results")
    stats['zones_tracked'] = c.fetchone()[0]

    c.execute("SELECT MIN(recorded_at), MAX(recorded_at) FROM sensor_readings")
    row = c.fetchone()
    stats['first_reading'] = row[0]
    stats['last_reading']  = row[1]

    c.execute("""
        SELECT COUNT(*) FROM risk_results
        WHERE urgency_category = 'Immediate Attention'
    """)
    stats['total_critical_events'] = c.fetchone()[0]

    conn.close()
    return stats


def clear_db():
    """Wipes all data. Use only for testing/reset."""
    conn = _connect()
    conn.execute("DELETE FROM sensor_readings")
    conn.execute("DELETE FROM risk_results")
    conn.commit()
    conn.close()
    print("[DB] All data cleared.")