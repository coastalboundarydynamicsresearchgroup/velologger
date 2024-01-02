import logging
from typing import List

import numpy as np


class GPRMCArray:
    def __init__(self, nmea_str_list: List[str]):
        self.nmea_str_list = nmea_str_list
        gprmc_list = [GPRMC(x) for x in nmea_str_list]
        self.datetime_utc = np.array(
            [np.datetime64(x.datetime_utc_str) if x.is_valid else np.nan for x in gprmc_list]
        ).astype("datetime64[us]")
        self.status = np.array([x.status if x.is_valid else np.nan for x in gprmc_list])
        self.latitude_deg = np.array([x.latitude_deg if x.is_valid else np.nan for x in gprmc_list])
        self.longitude_deg = np.array(
            [x.longitude_deg if x.is_valid else np.nan for x in gprmc_list]
        )
        self.speed = np.array([x.speed if x.is_valid else np.nan for x in gprmc_list])
        self.track_angle = np.array([x.track_angle if x.is_valid else np.nan for x in gprmc_list])
        self.magnetic_variation = np.array(
            [x.magnetic_variation if x.is_valid else np.nan for x in gprmc_list]
        )
        self.magnetic_variation_direction = np.array(
            [x.magnetic_variation_direction if x.is_valid else np.nan for x in gprmc_list]
        )

    def __repr__(self):
        return (
            f"GPRMCArray(datetime_utc={self.datetime_utc}, status={self.status}, "
            f"latitude={self.latitude_deg}, longitude={self.longitude_deg}, "
            f"speed={self.speed}, track_angle={self.track_angle}, "
            f"magnetic_variation={self.magnetic_variation}, "
            f"magnetic_variation_direction={self.magnetic_variation_direction})"
        )


class GPRMC:
    def __init__(self, nmea_sentence: str):
        split_sentence = nmea_sentence.split(",")

        # Check that the sentence is a GPRMC sentence
        if split_sentence[0] != "$GPRMC":
            logging.debug(f"Not a GPRMC sentence: '{nmea_sentence}'")
            self.is_valid = False
            return

        self.is_valid = True

        # Parse the UTC time. Note: this does not include the date.
        self.datetime_utc_str = self._parse_date_time(split_sentence[1], split_sentence[9])

        # Status: A=active, V=void (not valid)
        self.status = split_sentence[2]

        # Latitude: format is ddmm.mmmm
        self.latitude_deg = self._parse_latitude(split_sentence[3], split_sentence[4])

        # Longitude: format is dddmm.mmmm
        self.longitude_deg = self._parse_longitude(split_sentence[5], split_sentence[6])

        # Speed over ground in knots
        self.speed = float(split_sentence[7]) if split_sentence[7] else 0.0

        # Track angle in degrees
        self.track_angle = float(split_sentence[8]) if split_sentence[8] else 0.0

        # Magnetic Variation
        self.magnetic_variation = float(split_sentence[10]) if split_sentence[10] else 0.0
        self.magnetic_variation_direction = split_sentence[11] if split_sentence[11] else np.nan

    def _parse_latitude(self, latitude_str, direction):
        if not latitude_str:
            return None
        latitude_deg = int(latitude_str[:2])
        latitude_min = float(latitude_str[2:])
        latitude = latitude_deg + latitude_min / 60.0
        return latitude if direction == "N" else -latitude

    def _parse_longitude(self, longitude_str, direction):
        if not longitude_str:
            return None
        longitude_deg = int(longitude_str[:3])
        longitude_min = float(longitude_str[3:])
        longitude = longitude_deg + longitude_min / 60.0
        return longitude if direction == "E" else -longitude

    def _parse_date_time(self, time_str, date_str):
        if not time_str or not date_str:
            return None
        # Construct a datetime string in the format 'YYYY-MM-DDTHH:MM:SS'
        year = int(date_str[4:6]) + 2000  # Adjust the year according to your context
        month = int(date_str[2:4])
        day = int(date_str[0:2])
        hours = int(time_str[0:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        return f"{year:04d}-{month:02d}-{day:02d}T{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __repr__(self):
        return (
            f"GPRMC(datetime_utc_str={self.datetime_utc_str}, status={self.status}, "
            f"latitude={self.latitude}, longitude={self.longitude}, "
            f"speed={self.speed}, track_angle={self.track_angle}, "
            f"magnetic_variation={self.magnetic_variation}, "
            f"magnetic_variation_direction={self.magnetic_variation_direction})"
        )
