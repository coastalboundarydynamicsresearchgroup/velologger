# %%
from enum import Enum

import numpy as np

# repeated gyro, temperature, and acceleration data
dtype_gyro_temp_accel = np.dtype(
    [
        ("gyro", np.uint16),
        ("temperature", np.uint16),
        ("acceleration_x", np.uint16),
        ("acceleration_y", np.uint16),
    ]
)

# telemetry packet
dtype_telemetry_packet = np.dtype(
    [
        ("not_used_1", np.uint8, 14),
        ("gyro_temp_accel", dtype_gyro_temp_accel, 3),
        ("not_used_2", np.uint8, 160),
        ("gps_microseconds_from_hour", np.uint32),
        ("not_used_3", np.uint8, 4),
        ("nmea_sentence", np.uint8, 72),
        ("not_used_4", np.uint8, 234),
    ]
)


class TelemetryPacketConstants(Enum):
    GYRO_SCALE_FACTOR: float = 0.09766
    TEMP_SCALE_FACTOR: float = 0.1453
    TEMP_BIAS: float = 25.0
    ACCEL_SCALE_FACTOR: float = 0.001221
