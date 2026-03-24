import pandas as pd
import numpy as np
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None


# Zone assigned to this hardware node — can be overridden at runtime
DEFAULT_ZONE = "Live-Sensor-Node"


class SerialIOTReader:
    """
    Reads live Arduino sensor data over Serial.

    Arduino output format (CSV per line):
        timestamp_ms,flex_raw,ultrasonic_cm,accel_x,accel_y,accel_z

    Flex sensor     → deformation_mm (normalized)
    Ultrasonic      → gap_mm
    MPU6050 accel   → vibration_level (magnitude minus gravity)
    """

    def __init__(self, port, baud=9600, window_seconds=120, zone_id=DEFAULT_ZONE):
        self.port = port
        self.baud = baud
        self.window_seconds = window_seconds
        self.zone_id = zone_id
        self.ser = None
        self.buffer = []
        self.is_active = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self):
        if serial is None:
            raise ImportError("pyserial not installed. Run: pip install pyserial")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.ser.reset_input_buffer()
            self.is_active = True
            return True
        except Exception as e:
            self.is_active = False
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_active = False

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_sync(self):
        """Pull all available lines from serial buffer into internal buffer."""
        if not self.ser or not self.ser.is_open:
            return

        while self.ser.in_waiting:
            try:
                raw = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not raw or raw.startswith('#'):
                    continue

                parts = raw.split(',')
                if len(parts) < 6:
                    continue

                flex_raw = float(parts[1])
                ultra_cm = float(parts[2])
                ax = float(parts[3])
                ay = float(parts[4])
                az = float(parts[5])

                # Vibration: acceleration magnitude minus 1g (gravity)
                vibration = max(0.0, np.sqrt(ax**2 + ay**2 + az**2) - 1.0)

                self.buffer.append({
                    'timestamp': datetime.now(),
                    'zone_id': self.zone_id,
                    'deformation_mm': (flex_raw / 1023.0) * 60.0,  # 0–60 mm range
                    'gap_mm': ultra_cm,
                    'vibration_level': round(vibration, 4),
                    'temperature': 25.0,   # placeholder — no temp sensor yet
                    'load_estimate': 500.0, # placeholder — HX711 not yet integrated
                })

            except (ValueError, IndexError) as e:
                print(f"[SerialReader] Parse error: {e}")
                continue

        # Prune old readings
        now = datetime.now()
        self.buffer = [
            d for d in self.buffer
            if (now - d['timestamp']).total_seconds() <= self.window_seconds
        ]

    def get_dataframe(self):
        if not self.buffer:
            return None
        return pd.DataFrame(self.buffer)
