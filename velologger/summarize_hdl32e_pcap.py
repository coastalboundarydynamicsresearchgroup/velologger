# %%
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from hdl32e.read_hdl32e_pcap_pointcloud import read_hdl32e_pcap_pointcloud
from matplotlib.ticker import ScalarFormatter
from pytars.readers.maptiles.get_map_tiles import get_map_data_from_lat_lon
from pytars.readers.pcap.pcap_filters import PcapPacketFilters
from scipy.stats import binned_statistic_2d


def summarize_hdl32e_pcap(
    pcap_file: Union[Path, str],
    zoom_level: int = 18,
    source_ip_addr: Optional[str] = None,
    min_relative_time: float = 0.0,
    max_relative_time: float = np.inf,
):
    """Summarize the HDL32e pcap file.
    Args:
        pcap_file: path to pcap file
        zoom_level: zoom level for map
        source_ip_addr: source ip address to filter
        min_relative_time: minimum relative time to filter
        max_relative_time: maximum relative time to filter
    """

    pcap_path = Path(pcap_file)
    PCAP_FILTERS = PcapPacketFilters(
        source_ip_addr=source_ip_addr,
        relative_time_gate_seconds=(min_relative_time, max_relative_time),
    )

    pc = read_hdl32e_pcap_pointcloud(
        pcap_file=pcap_path,
        pcap_filters=PCAP_FILTERS,
        calibration_file=None,
        extrinsic_4x4_to_transformed_from_sensor=None,
        include_null_returns=False,
        include_telemetry=True,
        name=str(pcap_path.stem),
    )

    save_prefix = pcap_path.with_suffix("")

    # visualize the pointcloud in spherical coordinates
    fig = plot_pointcloud_spherical(pc)
    fig.savefig(str(save_prefix) + "_summary_spherical.png", dpi=300)

    # visualize the gps data on a map
    fig = plot_gps_map(pc, zoom_level=zoom_level)
    if fig is not None:
        fig.savefig(str(save_prefix) + "_summary_gps.png", dpi=300)


def plot_pointcloud_spherical(pc):
    # calculate the grid vectors
    azimuth_vector_degrees = np.arange(0, 360, 0.5)
    elevation_vector_degrees = np.arange(-32, 12, 1.5)
    delta_range = [0, 0.5]
    # calculate median range for each bin
    median_range_meters = binned_statistic_2d(
        pc.sensor_frame.azimuth_degrees,
        pc.sensor_frame.elevation_degrees,
        pc.sensor_frame.range_meters,
        statistic="median",
        bins=[azimuth_vector_degrees, elevation_vector_degrees],
    ).statistic

    # calculate median reflectivity for each bin
    median_reflectivity = binned_statistic_2d(
        pc.sensor_frame.azimuth_degrees,
        pc.sensor_frame.elevation_degrees,
        pc.reflectivity,
        statistic="median",
        bins=[azimuth_vector_degrees, elevation_vector_degrees],
    ).statistic

    # calculate stdiance of range for each bin
    std_range_meters = binned_statistic_2d(
        pc.sensor_frame.azimuth_degrees,
        pc.sensor_frame.elevation_degrees,
        pc.sensor_frame.range_meters,
        statistic="std",
        bins=[azimuth_vector_degrees, elevation_vector_degrees],
    ).statistic

    # calculate min range for each bin
    min_range_meters = binned_statistic_2d(
        pc.sensor_frame.azimuth_degrees,
        pc.sensor_frame.elevation_degrees,
        pc.sensor_frame.range_meters,
        statistic="min",
        bins=[azimuth_vector_degrees, elevation_vector_degrees],
    ).statistic

    # calculate max range for each bin
    max_range_meters = binned_statistic_2d(
        pc.sensor_frame.azimuth_degrees,
        pc.sensor_frame.elevation_degrees,
        pc.sensor_frame.range_meters,
        statistic="max",
        bins=[azimuth_vector_degrees, elevation_vector_degrees],
    ).statistic

    # visualize the pointcloud in spherical coordinates
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    plot_data = [
        median_range_meters % 10,
        median_reflectivity,
        std_range_meters,
        median_reflectivity - min_range_meters,
        max_range_meters - median_reflectivity,
    ]
    plot_titles = [
        "Median Range % 10 (m)",
        "Median Reflectivity",
        "Std Range (m)",
        "Med - Min Range (m)",
        "Max - Med Range (m)",
    ]
    cmap_list = ["viridis_r", "jet", "gnuplot", "gnuplot", "gnuplot"]
    vrange_list = [[0, 10], [0, 100], delta_range, delta_range, delta_range]
    for ax, data, title, vrange, cmap in zip(axs, plot_data, plot_titles, vrange_list, cmap_list):
        im = ax.pcolormesh(
            azimuth_vector_degrees,
            elevation_vector_degrees,
            data.T,
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=cmap,
        )
        ax.set_xlim([0, 360])
        ax.set_aspect("equal")
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_title(title)
        ax.label_outer()
        fig.colorbar(im, ax=ax, aspect=5)

    for ax in [axs[0], axs[1], axs[3], axs[4]]:
        ax.set_ylabel("")

    duration_s = np.round((pc.datetime[-1] - pc.datetime[0]).astype(float) / 1e6, 3)
    suptitle_name = f"{pc.name}\n{str(pc.datetime[0])[:-3]} ({duration_s} seconds)"
    plt.suptitle(suptitle_name)

    plt.tight_layout()
    return fig


def plot_gps_map(pc, zoom_level: int):
    # visualize the gps position on a tiled map
    mean_latitude_deg = pc.telemetry.gprmc.latitude_deg.mean()
    mean_longitude_deg = pc.telemetry.gprmc.longitude_deg.mean()
    if ~np.isnan(mean_latitude_deg) and ~np.isnan(mean_longitude_deg):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        rgb_map_data = get_map_data_from_lat_lon(
            mean_latitude_deg,
            mean_longitude_deg,
            zoom_level=zoom_level,
            image_buffer_x=1,
            image_buffer_y=1,
        )
        ax.imshow(rgb_map_data.rgb, extent=rgb_map_data.extent_lon_lat)
        ax.plot(pc.telemetry.gprmc.longitude_deg, pc.telemetry.gprmc.latitude_deg, "r-")
        ax.plot(mean_longitude_deg, mean_latitude_deg, "r*", markersize=20)

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")

        duration_s = np.round((pc.datetime[-1] - pc.datetime[0]).astype(float) / 1e6, 3)
        suptitle_name = f"{pc.name}\n{str(pc.datetime[0])[:-3]} ({duration_s} seconds)"
        ax.set_title(suptitle_name)

        plt.tight_layout()
        return fig
    else:
        print("No valid GPS data")
        return None


if __name__ == "__main__":
    # PCAP_FILE = r"C:\Users\Richie\Downloads\HDL32-V2_Tunnel.pcap"

    # summarize_hdl32e_pcap(PCAP_FILE)
    Fire(summarize_hdl32e_pcap)
