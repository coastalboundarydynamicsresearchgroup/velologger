# %%
import matplotlib.pyplot as plt
import numpy as np
from hdl32e.read_hdl32e_pcap_pointcloud import read_hdl32e_pcap_pointcloud
from matplotlib.ticker import ScalarFormatter
from pytars.readers.maptiles.get_map_tiles import get_map_data_from_lat_lon
from pytars.readers.pcap.pcap_filters import PcapPacketFilters
from pytars.transforms.transform import create_rotation_matrix_4x4

# constants to read the pcap file
PCAP_FILE = r"/Users/rslocum/Downloads/lidar_20240104_232931.pcap"
PCAP_FILTERS = PcapPacketFilters(relative_time_gate_seconds=(0, 5))  #
EXTRINSIC_4X4_TO_TRANSFORMED_FROM_SENSOR = None
# logging.basicConfig(level=logging.DEBUG)

ROLL = 91.19
PITCH = 1.29
extrinsic_4x4_to_transformed_from_sensor = create_rotation_matrix_4x4(0, 0, 0, PITCH, -ROLL, 90)
pc = read_hdl32e_pcap_pointcloud(
    pcap_file=PCAP_FILE,
    pcap_filters=PCAP_FILTERS,
    extrinsic_4x4_to_transformed_from_sensor=extrinsic_4x4_to_transformed_from_sensor,
    include_null_returns=False,
    name="data collect 01",
)
# x = y
# y = -x
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(pc.transformed_frame.x_meters, pc.transformed_frame.y_meters, ".")
axs[0, 0].set_aspect("equal")
axs[0, 0].set_title("x vs y")

axs[0, 1].plot(pc.transformed_frame.x_meters, pc.transformed_frame.z_meters, ".")
axs[0, 1].set_aspect("equal")
axs[0, 1].set_title("x vs z")

axs[1, 0].plot(pc.transformed_frame.y_meters, pc.transformed_frame.z_meters, ".")
axs[1, 0].set_aspect("equal")
axs[1, 0].set_title("y vs z")

# visualize the pointcloud in spherical coordinates
# %%
plot_ind = pc.frame_num < 20  # & (pc.laser_id >= 15) & (pc.laser_id <= 16)
pc_plot = pc[plot_ind]

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
im = ax.scatter(
    pc_plot.transformed_frame.x_meters,
    pc_plot.transformed_frame.y_meters,
    20,
    pc_plot.transformed_frame.z_meters,
    vmin=-10,
    vmax=10,
    cmap="terrain",
)
ax.plot(0, 0, "m.", markersize=30)
ax.set_xlim([-40, 60])
ax.set_ylim([-20, 10])
ax.set_aspect("equal")
cbar = fig.colorbar(im, ax=ax, aspect=5, label="Elevation (m)")
# %%
plot_ind = (pc.frame_num == 2) & (pc.laser_id == 14)
pc_plot = pc[plot_ind]

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.plot(pc_plot.transformed_frame.x_meters, pc_plot.transformed_frame.z_meters, ".", markersize=10)
ax.plot(-pc_plot.sensor_frame.y_meters, -pc_plot.sensor_frame.x_meters, "r.", markersize=10)
# ax.set_aspect("equal")
ax.plot(0, 0, "m.", markersize=30)
# %%
fig, axs = plt.subplots(2, 1, figsize=(16, 4))

im = axs[0].scatter(
    pc_plot.sensor_frame.azimuth_degrees,
    pc_plot.sensor_frame.elevation_degrees,
    3,
    pc_plot.sensor_frame.range_meters % 10,
    vmin=0,
    vmax=10,
    cmap="viridis_r",
)
fig.colorbar(im, ax=axs[0], aspect=5, label="Range % 10 (m)")
im = axs[1].scatter(
    pc_plot.sensor_frame.azimuth_degrees,
    pc_plot.sensor_frame.elevation_degrees,
    3,
    pc_plot.reflectivity,
    vmin=0,
    vmax=100,
    cmap="jet",
)
fig.colorbar(im, ax=axs[1], aspect=5, label="Reflectivity")
axs[1].set_xlabel("Azimuth (deg)")

for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlim([0, 360])
    ax.set_ylabel("Elevation (deg)")

# visualize the gps position on a tiled map
mean_latitude_deg = pc.telemetry.gprmc.latitude_deg.mean()
mean_longitude_deg = pc.telemetry.gprmc.longitude_deg.mean()
if ~np.isnan(mean_latitude_deg) and ~np.isnan(mean_longitude_deg):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    rgb_map_data = get_map_data_from_lat_lon(
        mean_latitude_deg,
        mean_longitude_deg,
        zoom_level=17,
        image_buffer_x=1,
        image_buffer_y=1,
    )

    ax.imshow(rgb_map_data.rgb, extent=rgb_map_data.extent_lon_lat)
    ax.plot(pc.telemetry.gprmc.longitude_deg, pc.telemetry.gprmc.latitude_deg, "r-")
    ax.plot(mean_longitude_deg, mean_latitude_deg, "r*")

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("GPS Position")
else:
    print("No valid GPS data")
# plot xy data
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(
    pc_plot.sensor_frame.x_meters,
    pc_plot.sensor_frame.y_meters,
    3,
    pc_plot.sensor_frame.z_meters,
    vmin=-0.5,
    vmax=0.5,
)
ax.set_aspect("equal")
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Top Down View")

# % look at packet delays latency
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
t_packets = pc.per_packet_data.pcap_packet_relative_time_seconds

ax.plot(np.diff(t_packets) * 1e3, ".-")
ax.set_ylim([0, 2])

ax.set_ylabel("Packet Delay (ms)")
ax.set_xlabel("Packet Number")
ax.set_title("Time between packets")
# % look at gps time from telemetry
# %
if ~np.isnan(mean_latitude_deg) and ~np.isnan(mean_longitude_deg):
    try:
        fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
        # timestamps_from_pcap =datetime64_to_timestamp_seconds(pc.per_packet_data.pcap_packet_datetime)

        if pc.telemetry is not None and pc.telemetry.is_any_valid_gprmc:
            axs[0].plot(pc.telemetry.datetime, pc.telemetry.datetime, ".")
            axs[0].set_title("gps time from rmc")
        axs[1].plot(
            pc.per_packet_data.pcap_packet_datetime,
            pc.per_packet_data.pcap_packet_datetime,
            ".",
        )
        axs[1].set_title("gps time from pcap")
        axs[2].plot(pc.datetime, pc.datetime, ".")
        axs[2].set_title("point cloud timestamp per point")
        for ax in axs:
            ax.label_outer()
            # rotate tick labels
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.set_xlabel("Time")
            ax.label_outer()
    except ConnectionError:
        print("No Internet Connection for Basemap")
else:
    print("No valid GPS data")

# %
fig, ax = plt.subplots(1, 1)
num_frames = np.max(pc.frame_num)
for i in range(int(num_frames)):
    ax.plot(i, np.sum(pc.frame_num == i), ".")

# %%
d_motor_az = np.diff(pc.per_block_data.motor_azimuth_deg[::12])
d_packet_time = np.diff(pc.per_packet_data.pcap_packet_relative_time_seconds)
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(d_motor_az % 360, ".-")
axs[0].set_ylim([0, 2])
axs[1].plot(d_packet_time * 1000, ".-")
axs[1].set_ylim([0, 1])
axs[2].set_xlabel("Packet Number")
axs[0].set_title("Delta Motor Az (deg)")
axs[1].set_title("Delta packet received (ms)")
axs[2].set_title("pcap packet time (s)")

axs[2].plot(pc.per_packet_data.pcap_packet_relative_time_seconds, ".-")
axs[2].set_ylim([15.3, 15.35])

axs[1].set_xlim([23200, 23300])
plt.tight_layout()
