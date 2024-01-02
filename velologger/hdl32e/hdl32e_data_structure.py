# %%
from enum import Enum

import numpy as np

# Per channel/transceiver data type
dtype_point = np.dtype(
    [
        ("distance", np.uint16),
        ("intensity", np.uint8),
    ]
)


# firing data
dtype_block = np.dtype(
    [
        ("laser_block_id", np.uint16),
        ("motor_azimuth", np.uint16),
        ("laser_measurements", (dtype_point, 32)),
    ]
)

# data packet
dtype_data_packet = np.dtype(
    [
        ("firing_data", dtype_block, 12),
        ("gps_timestamp", np.uint32),
        ("factory", np.uint16),
    ]
)


class DataPacketConstants(Enum):
    MOTOR_AZIMUTH_SCALAR: float = 0.01  # 0.01 degrees
    DISTANCE_SCALAR: float = 0.002  # 2 mm
