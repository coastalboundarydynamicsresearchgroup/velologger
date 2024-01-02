# %%
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pytars.readers.pcap.pcap_filters import PcapPacketFilters
from pytars.readers.pcap.pcap_reader import PcapReader
from pytars.sensors.lidar.pointcloud import LidarPointCloud
from pytars.transforms.cart2sph_coordinate_system import CoordinateSystem
from pytars.transforms.coordinates import Coordinates
from pytars.utils.timing import datetime64_to_timestamp_seconds, mean_datetime64

from .hdl32e_data_structure import DataPacketConstants, dtype_data_packet
from .read_hdl32e_calibration import Hdl32CalibrationConstants, read_hdl32e_calibration
from .read_hdl32e_pcap_telemetry import Hdl32eTelemetryData, read_hdl32e_pcap_telemetry


@dataclass
class Hdl32eRawData:
    laser_block_id: np.ndarray
    motor_azimuth_per_block_deg: np.ndarray
    range_meters_per_point: np.ndarray
    intensity_per_point: np.ndarray
    gps_timestamp_microseconds_per_packet: np.ndarray
    factory_per_packet: np.ndarray
    pcap_packet_datetime: np.ndarray
    pcap_packet_relative_time_seconds: np.ndarray
    pcap_packet_ind: np.ndarray


@dataclass
class Hdl32ePerPacketData:
    gps_timestamp_microseconds: np.ndarray
    pcap_packet_datetime: np.ndarray
    pcap_packet_relative_time_seconds: np.ndarray
    pcap_packet_ind: np.ndarray


@dataclass
class Hdl32ePerBlockData:
    motor_azimuth_deg: np.ndarray


class DatetimeSource(Enum):
    """Enum for the source of the timestamp."""

    LIDAR = "time is microsconds from unknown hour"
    GPS = "time infered from GPRMC gps timestamp in telemetry packets"
    PCAP = "time infered from pcap packet timestamps"


@dataclass
class Hdl32ePointcloud(LidarPointCloud):
    packet_num: Optional[np.ndarray] = None
    block_num: Optional[np.ndarray] = None
    telemetry: Optional[Hdl32eTelemetryData] = None
    per_packet_data: Optional[Hdl32ePerPacketData] = None
    per_block_data: Optional[Hdl32ePerBlockData] = None
    datetime_source: DatetimeSource = DatetimeSource.LIDAR

    def __getitem__(self, key) -> "Hdl32ePointcloud":
        # Call the base class __getitem__
        base_item = super().__getitem__(key)

        # Handle the Hdl32ePointcloud specific attributes
        packet_num = self.packet_num[key] if self.packet_num is not None else None
        block_num = self.block_num[key] if self.block_num is not None else None

        # per packet and per block are not indexed
        telemetry = self.telemetry
        per_packet_data = self.per_packet_data
        per_block_data = self.per_block_data
        datetime_source = self.datetime_source

        # Create a new Hdl32ePointcloud instance with both base and specific attributes
        return Hdl32ePointcloud(
            sensor_frame=base_item.sensor_frame,
            transformed_frame=base_item.transformed_frame,
            intensity=base_item.intensity,
            reflectivity=base_item.reflectivity,
            return_num=base_item.return_num,
            num_returns=base_item.num_returns,
            laser_id=base_item.laser_id,
            frame_num=base_item.frame_num,
            datetime=base_item.datetime,
            lidar_type=base_item.lidar_type,
            name=base_item.name,
            packet_num=packet_num,
            block_num=block_num,
            telemetry=telemetry,
            per_packet_data=per_packet_data,
            per_block_data=per_block_data,
            datetime_source=datetime_source,
        )


def calc_block_num(num_packets, is_dual_return):
    """Calculate block num for each firing."""
    # create block num and return num for each firing
    if is_dual_return:
        block_num = np.kron(np.tile(np.arange(6), (32, 1)), np.array([1, 1]))
    else:
        block_num = np.tile(np.arange(12), (32, 1))

    return np.tile(block_num, (1, num_packets))


def calc_return_num_and_num_returns(
    range_meters: np.ndarray, is_dual_return: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate return_num, num_returns, is_unique for each firing.

    - if dual return:
        is_unique is True for the first return, False if second return range is == first
    - else:
        is_unique is True for all returns
    """
    num_blocks = int(range_meters.shape[1])

    if is_dual_return:
        # using 0 and -1 because we add either 1 or 2 later
        # order is alwaus [LAST, Strongest/Second Strongest]
        return_num = np.tile(np.array([0, -1]), (32, num_blocks // 2))

        # determine if dual return
        is_different_ranges = range_meters[:, ::2] != range_meters[:, 1::2]
        is_second_block_zero = range_meters[:, 1::2] == 0  # weird edge case
        is_dual_return = is_different_ranges & ~is_second_block_zero

        # determine if no returns
        is_no_returns = (range_meters[:, ::2] == 0) & (range_meters[:, 1::2] == 0)
        is_no_returns = np.kron(is_no_returns, np.array([1, 1]))

        # determine number of returns and update return_num
        num_returns = np.kron(is_dual_return, np.array([1, 1])) + 1
        num_returns[is_no_returns == 1] = 0
        return_num += num_returns

        # mask second block values when not dual return
        is_unique_return = (~np.kron(~is_dual_return, np.array([0, 1]))).astype(bool)

    else:
        return_num = np.ones_like(range_meters, dtype=int)
        num_returns = np.ones_like(range_meters, dtype=int)
        is_unique_return = np.ones_like(range_meters, dtype=bool)

    return return_num, num_returns, is_unique_return


def calc_azimuth_elevation_deg_time_usec(
    calibration, laser_id, block_num, motor_azimuth_deg, is_dual_return
):
    """Calculate azimuth, elevation, and packet time offset for each point."""
    # calculate timing offset
    packet_time_offset_microseconds = calibration.calc_timing_offset_microseconds(
        laser_id, block_num
    )
    # calculate relative to first firing in each block
    relative_time_offset_microseconds = (
        packet_time_offset_microseconds - packet_time_offset_microseconds[0, :]
    )

    # calculate motor speed (deg/s) for interpolation
    # this is weird because we're interpolating between firings -
    # the 12th firing can't interpolate to the first of the next packet because there's no
    # guarantee that the motor has moved to the next firing position
    # so we only interpolate between blocks in the same packet

    if is_dual_return:
        motor_speed_deg_per_usec = np.zeros(len(motor_azimuth_deg) // 2)
        motor_speed_deg_per_usec[:-1] = (
            np.diff(motor_azimuth_deg[::2]) % 360
        ) / Hdl32CalibrationConstants.BLOCK_TIME_MICROSECONDS.value
        # extrapolating
        motor_speed_deg_per_usec[5::6] = motor_speed_deg_per_usec[4::6]
        motor_speed_deg_per_usec = np.kron(motor_speed_deg_per_usec, np.array([1, 1]))
    else:
        motor_speed_deg_per_usec = np.zeros(len(motor_azimuth_deg))
        motor_speed_deg_per_usec[:-1] = (
            np.diff(motor_azimuth_deg) % 360
        ) / Hdl32CalibrationConstants.BLOCK_TIME_MICROSECONDS.value
        # extrapolating
        motor_speed_deg_per_usec[11::6] = motor_speed_deg_per_usec[10::6]

    # calculate azimuth angle
    relative_azimuth_degrees = (
        relative_time_offset_microseconds * motor_speed_deg_per_usec
    )
    azimuth_degrees = (
        motor_azimuth_deg
        + relative_azimuth_degrees
        + calibration.azimuth_offset_deg[laser_id]
    ) % 360
    elevation_degrees = calibration.elevation_offset_deg[laser_id]
    return azimuth_degrees, elevation_degrees, packet_time_offset_microseconds


def read_raw_hdl32e_pcap_data(pcap_file, pcap_filters: PcapPacketFilters):
    """Reads the raw data from a pcap file.

    - Includes all data from the udp packet.
    - Only includes some data from the pcap packet header.
    """
    # unless the user specifically requests a different destination port, use the default
    pcap_filters = pcap_filters.copy()
    if pcap_filters.destination_port is None:
        pcap_filters.destination_port = 2368
    pcap_filters.udp_payload_length_gate = [1206, 1206]

    # read all the packets
    with PcapReader(pcap_file, packet_filters=pcap_filters) as pcap:
        logging.debug(f"Reading HDL32e Pointcloud Packets: {str(pcap_file)}")
        all_udp_data = []
        all_packet_headers = []
        all_packet_ind = []
        for packet in pcap:
            udp_data = packet.udp_data.data
            all_udp_data.append(udp_data)
            all_packet_headers.append(packet.packet_header)
            all_packet_ind.append(packet.packet_ind)

    # combine all the packets into one byte array
    combined_bytes = b"".join(all_udp_data)

    # parse the byte array into a numpy array
    data = np.frombuffer(combined_bytes, dtype=dtype_data_packet)

    # check that data exists
    if len(data) == 0:
        raise ValueError("No data found in pcap file")

    # extract the data
    laser_block_id = np.hstack([x["firing_data"]["laser_block_id"] for x in data])
    motor_azimuth_deg = (
        np.hstack([x["firing_data"]["motor_azimuth"] for x in data])
        * DataPacketConstants.MOTOR_AZIMUTH_SCALAR.value
    )

    range_meters = (
        np.vstack([x["firing_data"]["laser_measurements"]["distance"] for x in data])
        * DataPacketConstants.DISTANCE_SCALAR.value
    ).T
    intensity = np.vstack(
        [x["firing_data"]["laser_measurements"]["intensity"] for x in data]
    ).T

    gps_timestamp_microseconds = np.hstack([x["gps_timestamp"] for x in data])

    factory_per_packet = np.hstack([x["factory"] for x in data])

    return Hdl32eRawData(
        laser_block_id=laser_block_id,
        motor_azimuth_per_block_deg=motor_azimuth_deg,
        range_meters_per_point=range_meters,
        intensity_per_point=intensity,
        gps_timestamp_microseconds_per_packet=gps_timestamp_microseconds,
        factory_per_packet=np.array(factory_per_packet),
        pcap_packet_datetime=np.array([x.datetime for x in all_packet_headers]).astype(
            "datetime64[us]"
        ),
        pcap_packet_relative_time_seconds=np.array(
            [x.relative_timestamp_seconds for x in all_packet_headers]
        ),
        pcap_packet_ind=np.array(all_packet_ind),
    )


def correct_lidar_time(
    lidar_time_seconds: np.ndarray,
    raw_telemetry,
    per_packet_data: Optional[Hdl32ePerPacketData] = None,
) -> Tuple[np.ndarray, DatetimeSource]:
    """Correct the lidar time based on the telemetry.

    There has to be a more elegant way to do this - packet structure is very inefficient

    time from the lidar is microseconds from the top of the hour
    there is no data in the udp packets to inform what the hour is
    telemetry packets record full gps time
    we can use the gps time to infer the hour
     - the edge case is when the data spans an hour change - out median hour could be off by 1
     - this assumes the data doesn't span more than 1 hour
    if telemetry is not avaiable or has no gps, we fall back to the computer time in the pcap
     - but even this has flaws, the mean pcap time is used to determine the closest hour
     - if the time in the packet (gps_time_microseconds) is way off, so is the timestamp
    """
    # determine whether to use gps time or pcap computer time
    # check if raw telemetry is not None and telemetry datetime gprmc time is not all nan

    if raw_telemetry is not None and raw_telemetry.is_any_valid_gprmc:
        metadata_datetime = raw_telemetry.datetime
        datetime_source = DatetimeSource.GPS
    else:
        metadata_datetime = per_packet_data.pcap_packet_datetime
        datetime_source = DatetimeSource.PCAP
        logging.debug("Using pcap packet time for timestamp")
    logging.debug(f"  min telemetry datetime: {metadata_datetime[0]}")
    logging.debug(f"  max telemetry datetime: {metadata_datetime[-1]}")

    # ensure data coverage doesn't span more than an hour
    time_coverage = (metadata_datetime[0] - metadata_datetime[-1]).astype(
        "timedelta64[h]"
    )
    if time_coverage > 1:
        logging.error(
            "Can't convert time to datetime: time conversion assumes less than 1 hour of data"
        )
        return (lidar_time_seconds * 1e6).astype("datetime64[us]"), DatetimeSource.LIDAR

    # round to low, mid, high hour and add lidar time from top of hour
    mean_datetime = mean_datetime64(metadata_datetime)
    low_estimate_datetime = (mean_datetime.astype("datetime64[h]") - 1).astype(
        "datetime64[us]"
    ) + (lidar_time_seconds * 1e6).astype("timedelta64[us]")

    mid_estimate_datetime = mean_datetime.astype("datetime64[h]").astype(
        "datetime64[us]"
    ) + (lidar_time_seconds * 1e6).astype("timedelta64[us]")
    high_estimate_datetime = (mean_datetime.astype("datetime64[h]") + 1).astype(
        "datetime64[us]"
    ) + (lidar_time_seconds * 1e6).astype("timedelta64[us]")
    # calculate the difference between the time estimates and the mean_datetime
    all_estimates = np.stack(
        [low_estimate_datetime, mid_estimate_datetime, high_estimate_datetime]
    )
    all_estimate_diff = np.abs(all_estimates - mean_datetime)

    # determine which estimate is closest to the median
    ind = np.argmin(all_estimate_diff, axis=0)
    # use the closest estimate
    corrected_lidar_datetime = (lidar_time_seconds * 1e6).astype("datetime64[us]")
    corrected_lidar_datetime[ind == 0] = low_estimate_datetime[ind == 0]
    corrected_lidar_datetime[ind == 1] = mid_estimate_datetime[ind == 1]
    corrected_lidar_datetime[ind == 2] = high_estimate_datetime[ind == 2]

    return corrected_lidar_datetime, datetime_source


def read_hdl32e_pcap_pointcloud(
    pcap_file,
    pcap_filters: PcapPacketFilters,
    calibration_file: Optional[Union[Path, str]] = None,
    extrinsic_4x4_to_transformed_from_sensor: Optional[np.ndarray] = None,
    include_null_returns: bool = False,
    include_per_packet_per_block_data: bool = True,
    include_telemetry: bool = True,
    name: str = "",
):
    raw_data = read_raw_hdl32e_pcap_data(pcap_file, pcap_filters)
    if include_telemetry:
        raw_telemetry = read_hdl32e_pcap_telemetry(pcap_file, pcap_filters)
    else:
        raw_telemetry = None

    # read calibration
    calibration = read_hdl32e_calibration(calibration_file)

    # infer if dual return based on first packet
    is_dual_return_per_point = np.all(
        raw_data.motor_azimuth_per_block_deg[0:12:2]
        == raw_data.motor_azimuth_per_block_deg[1:12:2]
    )

    # calculate return_num and num_returns
    (
        return_num_per_point,
        num_returns_per_point,
        is_unique_return_per_point,
    ) = calc_return_num_and_num_returns(
        raw_data.range_meters_per_point, is_dual_return_per_point
    )

    # calculate null returns
    is_null_return_per_point = raw_data.range_meters_per_point == 0

    num_blocks = int(len(raw_data.motor_azimuth_per_block_deg))
    num_packets = int(num_blocks / 12)
    laser_id_per_point = np.tile(np.arange(32), (num_blocks, 1)).T
    block_num_per_point = calc_block_num(num_packets, is_dual_return_per_point)

    (
        azimuth_degrees_per_point,
        elevation_degrees_per_point,
        packet_time_offset_microseconds_per_point,
    ) = calc_azimuth_elevation_deg_time_usec(
        calibration,
        laser_id_per_point,
        block_num_per_point,
        raw_data.motor_azimuth_per_block_deg,
        is_dual_return_per_point,
    )

    # convert offset to absolute time
    gps_timestamp_microseconds_per_block = np.kron(
        raw_data.gps_timestamp_microseconds_per_packet, np.ones(12)
    )

    lidar_time_seconds = (
        packet_time_offset_microseconds_per_point + gps_timestamp_microseconds_per_block
    ) * 1e-6

    # calculate frame_num
    motor_direction = np.median(np.diff(raw_data.motor_azimuth_per_block_deg[::2]))
    frame_num_per_packet = np.zeros(num_blocks)
    if motor_direction > 0:
        frame_num_per_packet[1:] = np.cumsum(
            np.diff(raw_data.motor_azimuth_per_block_deg) < 0
        )
        logging.debug("Motor direction is positive")
    else:
        frame_num_per_packet[1:] = np.cumsum(
            np.diff(raw_data.motor_azimuth_per_block_deg) > 0
        )
        logging.debug("Motor direction is negative")
    frame_num_per_point = np.tile(frame_num_per_packet, (32, 1))

    # create coordinates class
    coordinate_system = CoordinateSystem(
        is_cartesian_right_handed=True,
        is_spherical_right_handed=False,
        is_elevation_0_horizon=True,
        azimuth_0_right_hand_math_degrees=90,
    )
    az_el_deg_range_meters = np.stack(
        [
            azimuth_degrees_per_point.T.flatten(),
            elevation_degrees_per_point.T.flatten(),
            raw_data.range_meters_per_point.T.flatten(),
        ],
        axis=1,
    )

    sensor_frame = Coordinates(
        coordinate_system=coordinate_system,
        az_el_deg_range_meters=az_el_deg_range_meters,
    )

    # apply extrinsic rotation
    if extrinsic_4x4_to_transformed_from_sensor is None:
        transformed_frame = None
    else:
        transformed_frame = sensor_frame.transformed(
            extrinsic_4x4_to_transformed_from_sensor
        )

    if include_per_packet_per_block_data:
        # create per block class
        per_block = Hdl32ePerBlockData(
            motor_azimuth_deg=raw_data.motor_azimuth_per_block_deg,
        )

        # create per packet class
        per_packet = Hdl32ePerPacketData(
            gps_timestamp_microseconds=raw_data.gps_timestamp_microseconds_per_packet,
            pcap_packet_datetime=raw_data.pcap_packet_datetime,
            pcap_packet_relative_time_seconds=raw_data.pcap_packet_relative_time_seconds,
            pcap_packet_ind=raw_data.pcap_packet_ind,
        )
    else:
        per_block = None
        per_packet = None

    # update timestamp based on source
    corrected_lidar_datetime, datetime_source = correct_lidar_time(
        lidar_time_seconds, raw_telemetry, per_packet
    )
    # create Hdl32ePointcloud class
    pc = Hdl32ePointcloud(
        sensor_frame=sensor_frame,
        transformed_frame=transformed_frame,
        intensity=None,
        reflectivity=raw_data.intensity_per_point.T.flatten(),
        return_num=return_num_per_point.T.flatten(),
        num_returns=num_returns_per_point.T.flatten(),
        laser_id=laser_id_per_point.T.flatten(),
        frame_num=frame_num_per_point.T.flatten(),
        datetime=corrected_lidar_datetime.T.flatten(),
        lidar_type="hdl32e",
        name=name,
        packet_num=(block_num_per_point // 12).T.flatten(),
        block_num=block_num_per_point.T.flatten(),
        telemetry=raw_telemetry,
        per_packet_data=per_packet,
        per_block_data=per_block,
        datetime_source=datetime_source,
    )

    # remove null if requested
    if not include_null_returns:
        pc = pc[
            ~is_null_return_per_point.T.flatten()
            & is_unique_return_per_point.T.flatten()
        ]
    else:
        pc = pc[is_unique_return_per_point.T.flatten()]

    return pc


if __name__ == "__main__":
    import time

    start_time = time.time()
    PCAP_FILE = "/Users/rslocum/Downloads/SampleData_600rpm_GPS.pcap"
    PCAP_FILE = "/Users/rslocum/Downloads/SampleData.pcap"

    PCAP_FILTERS = PcapPacketFilters(
        source_ip_addr="192.168.53.201",
        relative_time_gate_seconds=(0.0, 1),
    )
    EXTRINSIC_4X4_TO_TRANSFORMED_FROM_SENSOR = None
    # logging.basicConfig(level=logging.DEBUG)
    pc = read_hdl32e_pcap_pointcloud(
        pcap_file=PCAP_FILE,
        pcap_filters=PCAP_FILTERS,
        calibration_file=None,
        extrinsic_4x4_to_transformed_from_sensor=EXTRINSIC_4X4_TO_TRANSFORMED_FROM_SENSOR,
        include_null_returns=False,
        name="test",
    )
    time_to_read = time.time() - start_time

    print(f"Number of points: {len(pc)}")
    dt_seconds = (pc.datetime[-1] - pc.datetime[0]).astype("timedelta64[us]").astype(
        float
    ) / 1e6
    print(f"Duration read (seconds): {dt_seconds}")
    print(f"Percent of sensor time to read: {time_to_read/(dt_seconds) * 100:.1f}%")

    # %
    import matplotlib.pyplot as plt

    pc2 = pc[pc.frame_num == 1]
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))

    im = axs[0].scatter(
        (pc2.sensor_frame.azimuth_degrees + 180) % 360 - 180,
        pc2.sensor_frame.elevation_degrees,
        3,
        pc2.sensor_frame.range_meters,
        vmin=0,
        vmax=5,
        cmap="viridis_r",
    )
    fig.colorbar(im, ax=axs[0], label="Range (m)")
    im = axs[1].scatter(
        (pc2.sensor_frame.azimuth_degrees + 180) % 360 - 180,
        pc2.sensor_frame.elevation_degrees,
        3,
        pc2.reflectivity,
        vmin=0,
        vmax=100,
        cmap="jet",
    )
    fig.colorbar(im, ax=axs[1], label="Reflectivity")
    axs[1].set_xlabel("Azimuth (deg)")
    axs[0].set_title("Range (m)")
    axs[1].set_title("Reflectivity")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim([-170, 80])

    # %
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    pc_plot = pc[pc.frame_num == 2]
    ax.scatter(
        pc_plot.sensor_frame.x_meters,
        pc_plot.sensor_frame.y_meters,
        1,
        pc_plot.sensor_frame.z_meters,
    )
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 2])
    # %
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    if pc.telemetry is not None:
        timestamps_from_gps = datetime64_to_timestamp_seconds(pc.telemetry.datetime)
        print(
            f"{'gps time from telemetry':<30}:"
            + f"{pc.telemetry.datetime[0]} {pc.telemetry.datetime[-1]}"
        )
    else:
        timestamps_from_gps = np.array([0]) * np.nan
    timestamps_from_pcap = datetime64_to_timestamp_seconds(
        pc.per_packet_data.pcap_packet_datetime
    )
    print(
        f"{'pcap time from packet':<30}:"
        + f"{pc.per_packet_data.pcap_packet_datetime[0]}"
        + f"{pc.per_packet_data.pcap_packet_datetime[-1]}"
    )
    print(f"{'point cloud time':<30}: {pc.datetime[0]} {pc.datetime[-1]}")

    axs[0].plot(timestamps_from_gps, timestamps_from_gps, ".")
    axs[0].set_title("gps time from rmc")
    axs[1].plot(timestamps_from_pcap, timestamps_from_pcap, ".")
    axs[1].set_title("gps time from pcap")
    axs[2].plot(pc.timestamp_seconds, pc.timestamp_seconds, ".")
    axs[2].set_title("point cloud timestamp per point")
    for ax in axs:
        ax.label_outer()
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
