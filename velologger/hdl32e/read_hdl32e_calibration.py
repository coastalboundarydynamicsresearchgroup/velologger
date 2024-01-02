# %%
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np


class Hdl32CalibrationConstants(Enum):
    BLOCK_TIME_MICROSECONDS: float = 46.08
    CHANNEL_TIME_MICROSECONDS: float = 1.152


@dataclass
class Hdl32eCalibration:
    elevation_offset_deg: np.ndarray
    azimuth_offset_deg: np.ndarray
    file_name: str

    @staticmethod
    def calc_timing_offset_microseconds(
        laser_id: np.ndarray, block_num: np.ndarray
    ) -> np.ndarray:
        return (
            Hdl32CalibrationConstants.BLOCK_TIME_MICROSECONDS.value * block_num
            + Hdl32CalibrationConstants.CHANNEL_TIME_MICROSECONDS.value * laser_id
        )


def read_hdl32e_calibration(file_name: Optional[Union[Path, str]] = None):
    default_file_path = Path(__file__).parent / "calibration/default_calibration.csv"
    if file_name is None:
        file_path = default_file_path
        return read_calibration_intrinsics_csv(file_path)
    elif Path(file_name).suffix == ".csv":
        return read_calibration_intrinsics_csv(file_name)
    else:
        raise NotImplementedError("Only csv calibration files are supported")


def read_calibration_intrinsics_csv(file_name: Union[Path, str]) -> Hdl32eCalibration:
    """Reads the calibration file for the HDL32e."""
    with open(file_name, "r") as f:
        lines = f.readlines()
        # format is laser_id, azimuth_offset_deg, elevation_offset_deg
        # skip header
        lines = lines[1:]
        # split lines
        lines = [line.split(",") for line in lines]
        # convert to floats
        lines = [[float(item) for item in line] for line in lines]
        # convert to numpy array
        lines = np.array(lines)
        # split into columns
        laser_id = lines[:, 0]
        # check that laser id is unique fro 0-31 in order
        if not np.all(laser_id == np.arange(32)):
            raise ValueError("Laser ID must be unique from 0-31 in order")

        azimuth_offset_deg = lines[:, 1]
        elevation_offset_deg = lines[:, 2]
        return Hdl32eCalibration(
            elevation_offset_deg=elevation_offset_deg,
            azimuth_offset_deg=azimuth_offset_deg,
            file_name=file_name,
        )
    pass
