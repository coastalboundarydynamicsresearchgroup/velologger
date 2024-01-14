# %%
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from pytars.readers.pcap.pcap_filters import PcapPacketFilters
from pytars.readers.pcap.pcap_reader import PcapReader

from .gnssnmea.gprmc import GPRMCArray
from .hdl32e_telemetry_structure import TelemetryPacketConstants, dtype_telemetry_packet


# define the data structure for the telemetry packets
@dataclass
class GyroTempAccel:
    gyro_deg_per_sec: np.ndarray
    temperature_celcius: np.ndarray
    acceleration_x_g: np.ndarray
    acceleration_y_g: np.ndarray


@dataclass
class TelemetryPcapPacketsData:
    timestamp_sec: np.ndarray
    relative_time_sec: np.ndarray
    datetime: np.ndarray
    packet_inds: np.ndarray


@dataclass
class Hdl32eTelemetryData:
    raw_gyro_temp_accel: List[GyroTempAccel]
    gps_microseconds_from_hour: np.ndarray
    gprmc: GPRMCArray
    pcap_packet: TelemetryPcapPacketsData

    @property
    def is_any_valid_gprmc(self):
        return np.any(~np.isnat(self.gprmc.datetime_utc))

    @property
    def datetime(self):
        # assume microseconds is more accurate than gprmc
        datetime_truncated_hour = self.gprmc.datetime_utc.astype("datetime64[h]").astype(
            "datetime64[us]"
        )
        return datetime_truncated_hour + self.gps_microseconds_from_hour.astype("timedelta64[us]")

    @property
    def mean_accel_x(self):
        """Returns the average of the two accelerations in the x dimension."""
        accel_y_1 = self.raw_gyro_temp_accel[0].acceleration_y_g
        accel_x_3 = -self.raw_gyro_temp_accel[2].acceleration_x_g
        return (accel_y_1 - accel_x_3) / 2

    @property
    def mean_accel_y(self):
        """Returns the average of the two accelerations in the y dimension."""
        accel_y_2 = self.raw_gyro_temp_accel[1].acceleration_y_g
        accel_y_3 = self.raw_gyro_temp_accel[2].acceleration_y_g
        return (accel_y_2 + accel_y_3) / 2

    @property
    def mean_accel_z(self):
        """Returns the average of the two accelerations in the x dimension."""
        accel_x_1 = self.raw_gyro_temp_accel[0].acceleration_x_g
        accel_x_2 = self.raw_gyro_temp_accel[1].acceleration_x_g
        return (-accel_x_1 - accel_x_2) / 2

    @property
    def gyro_x(self):
        """Returns the gyro about the x axis using right hand convention."""
        return -self.raw_gyro_temp_accel[1].gyro_deg_per_sec

    @property
    def gyro_y(self):
        """Returns the gyro about the y axis using right hand convention."""
        return self.raw_gyro_temp_accel[0].gyro_deg_per_sec

    @property
    def gyro_z(self):
        """Returns the gyro about the z axis using right hand convention."""
        return self.raw_gyro_temp_accel[2].gyro_deg_per_sec


def read_hdl32e_pcap_telemetry(pcap_file, pcap_filters: PcapPacketFilters) -> Hdl32eTelemetryData:
    # unless the user specifically requests a different destination port, use the default
    pcap_filters = pcap_filters.copy()
    if pcap_filters.destination_port is None:
        pcap_filters.destination_port = 8308
    pcap_filters.udp_payload_length_gate = [512, 512]

    with PcapReader(pcap_file, packet_filters=pcap_filters) as pcap:
        logging.debug(f"Reading HDL32e Telemetry Packets: {str(pcap_file)}")
        all_udp_data = []
        all_packet_headers = []
        all_pcap_packet_inds = []
        for packet in pcap:
            udp_data = packet.udp_data.data
            all_udp_data.append(udp_data)
            all_packet_headers.append(packet.packet_header)
            all_pcap_packet_inds.append(packet.packet_ind)

    # combine all the packets into one byte array
    combined_bytes = b"".join(all_udp_data)

    if len(combined_bytes) == 0:
        logging.warning("No telemetry packets found in pcap file")
        return None

    # parse the byte array into a numpy array
    data = np.frombuffer(combined_bytes, dtype=dtype_telemetry_packet)

    # extra the gyro, temperature, and acceleration data
    all_gyro_temp_accel = []
    for i in range(3):
        i_gyro = (
            uint12_to_int12(np.array([x["gyro_temp_accel"][i]["gyro"] for x in data]) & 0x0FFF)
            * TelemetryPacketConstants.GYRO_SCALE_FACTOR.value
        )
        i_temp = (
            (
                uint12_to_int12(
                    np.array([x["gyro_temp_accel"][i]["temperature"] for x in data]) & 0x0FFF
                )
            )
            * TelemetryPacketConstants.TEMP_SCALE_FACTOR.value
            + TelemetryPacketConstants.TEMP_BIAS.value
        )
        i_accel_x = (
            uint12_to_int12(
                np.array([x["gyro_temp_accel"][i]["acceleration_x"] for x in data]) & 0x0FFF
            )
        ) * TelemetryPacketConstants.ACCEL_SCALE_FACTOR.value
        i_accel_y = (
            uint12_to_int12(
                np.array([x["gyro_temp_accel"][i]["acceleration_y"] for x in data]) & 0x0FFF
            )
        ) * TelemetryPacketConstants.ACCEL_SCALE_FACTOR.value
        all_gyro_temp_accel.append(GyroTempAccel(i_gyro, i_temp, i_accel_x, i_accel_y))
    all_gprmc = GPRMCArray([array_to_str(x["nmea_sentence"]) for x in data])
    all_gps_gps_microseconds_from_hour = np.array([x["gps_microseconds_from_hour"] for x in data])
    return Hdl32eTelemetryData(
        raw_gyro_temp_accel=all_gyro_temp_accel,
        gps_microseconds_from_hour=all_gps_gps_microseconds_from_hour,
        gprmc=all_gprmc,
        pcap_packet=TelemetryPcapPacketsData(
            np.array([x.timestamp_seconds for x in all_packet_headers]),
            np.array([x.relative_timestamp_seconds for x in all_packet_headers]),
            np.array([x.datetime for x in all_packet_headers]).astype("datetime64[us]"),
            np.array(all_pcap_packet_inds),
        ),
    )


def uint12_to_int12(uint12_value: np.ndarray):
    int12_value = uint12_value.copy().astype(np.int16)
    int12_value[int12_value > 2047] = int12_value[int12_value > 2047] - 4096
    return int12_value


def array_to_str(array: np.ndarray):
    return "".join(chr(byte) for byte in array)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    PCAP_NAME = "/Users/rslocum/Downloads/SampleData_600rpm_GPS.pcap"
    # PCAP_NAME = "/Users/rslocum/Downloads/SampleData.pcap"

    FILTERS = PcapPacketFilters(
        # max_packets=400,
        source_ip_addr="192.168.53.201",
        relative_time_gate_seconds=[0.0, 10],
    )
    # logging.basicConfig(level=logging.DEBUG)

    x = read_hdl32e_pcap_telemetry(PCAP_NAME, FILTERS)

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
    axs[0].plot(x.pcap_packet.relative_time_sec, x.mean_accel_x, label="mean accel x")
    axs[0].plot(x.pcap_packet.relative_time_sec, x.mean_accel_y, label="mean accel y")
    axs[0].plot(x.pcap_packet.relative_time_sec, x.mean_accel_z, label="mean accel z")
    axs[0].legend()
    axs[1].plot(x.pcap_packet.relative_time_sec, x.gyro_x, label="gyro x")
    axs[1].plot(x.pcap_packet.relative_time_sec, x.gyro_y, label="gyro y")
    axs[1].plot(x.pcap_packet.relative_time_sec, x.gyro_z, label="gyro z")
    axs[1].legend()
    axs[2].plot(
        x.pcap_packet.relative_time_sec,
        x.raw_gyro_temp_accel[0].temperature_celcius,
        label="temp 0",
    )
    axs[2].plot(
        x.pcap_packet.relative_time_sec,
        x.raw_gyro_temp_accel[1].temperature_celcius,
        label="temp 1",
    )
    axs[2].plot(
        x.pcap_packet.relative_time_sec,
        x.raw_gyro_temp_accel[2].temperature_celcius,
        label="temp 2",
    )
    axs[2].legend()

    axs[3].plot(x.pcap_packet.relative_time_sec, x.gprmc.latitude_deg)
    axs[4].plot(x.pcap_packet.relative_time_sec, x.gprmc.longitude_deg)

    # Create a ScalarFormatter object
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    for titlestr, y_str, ax in zip(
        ["acceleration", "gyro", "temperature", "latitude", "longitude"],
        ["$m/s^2$", "deg/sec", "deg Celcius", "Degrees", "Degrees"],
        axs,
    ):
        ax.set_title(titlestr)
        ax.set_xlabel("time (sec)")
        ax.set_ylabel(y_str)
        ax.label_outer()
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    # align y labels
    fig.align_ylabels(axs)

    fig, ax = plt.subplots(1, 1)
    dt_ms = np.diff(x.pcap_packet.relative_time_sec) * 1000
    rolling_mean = np.convolve(dt_ms, np.ones(20) / 20, mode="valid")
    ax.plot(dt_ms, ".")
    ax.plot(rolling_mean, "-")

    ax.set_title("time between telemetry packets")
    ax.set_xlabel("packet index")
    ax.set_ylabel("time (ms)")
