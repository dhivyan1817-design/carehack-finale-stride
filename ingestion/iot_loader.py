import os
import numpy as np
import pandas as pd
from database.db_manager import init_db, insert_readings

# Global reader instance (persists across Streamlit reruns)
_serial_reader = None


def get_serial_reader(port=None, baud=9600, zone_id="Live-Sensor-Node"):
    global _serial_reader
    if port and (_serial_reader is None or _serial_reader.port != port):
        from .serial_iot_reader import SerialIOTReader
        _serial_reader = SerialIOTReader(port, baud, zone_id=zone_id)
    return _serial_reader


def is_iot_connected() -> bool:
    global _serial_reader
    return _serial_reader is not None and _serial_reader.is_active


def load_iot_data():
    """
    Returns the latest buffered DataFrame from the active serial reader.
    Also persists new readings to the database.
    Returns None if not connected or buffer is empty.
    """
    global _serial_reader
    if not _serial_reader or not _serial_reader.is_active:
        return None

    _serial_reader.read_sync()
    df = _serial_reader.get_dataframe()

    # Persist to DB (fire-and-forget — never blocks the dashboard)
    if df is not None and not df.empty:
        try:
            insert_readings(df, data_source="iot")
        except Exception as e:
            print(f"[IoT Loader] DB write error: {e}")

    return df