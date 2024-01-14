# %%
import matplotlib.pyplot as plt
import numpy as np
from hdl32e.read_hdl32e_pcap_pointcloud import read_hdl32e_pcap_pointcloud
from pytars.readers.pcap.pcap_filters import PcapPacketFilters
from pytars.utils.gridding import bin_grid_1d, bin_grid_2d

# constants to read the pcap file
PCAP_FILE = r"/Users/rslocum/Downloads/lidar_20240104_232931.pcap"
PCAP_FILTERS = PcapPacketFilters(relative_time_gate_seconds=(0, 10))
# ^ setting values here can make it faster to debug if you only want to load a subset of the data

# Visual constants
DX = 0.25
DZ_CMAP_SCALE = 0.5
FRAME_NUM = 3

# read point cloud
pc = read_hdl32e_pcap_pointcloud(
    pcap_file=PCAP_FILE,
    pcap_filters=PCAP_FILTERS,  # This is optional, it can reduce the amount of data loaded
    name="sample collect",
)


# compute min z_grid
xi = np.arange(-50, 40, 0.5)
yi = np.arange(-10, 20, 0.5)
zg_mean = bin_grid_2d(
    pc.transformed_frame.x_meters,
    pc.transformed_frame.y_meters,
    pc.transformed_frame.z_meters,
    xi,
    yi,
    "min",
)

pc_frame = pc[pc.frame_num == FRAME_NUM]
zg_frame = bin_grid_2d(
    pc_frame.transformed_frame.x_meters,
    pc_frame.transformed_frame.y_meters,
    pc_frame.transformed_frame.z_meters,
    xi,
    yi,
    "mean",
)
# visualize the instantaneous z - mean z
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.pcolormesh(xi, yi, zg_mean, cmap="terrain", vmin=-10, vmax=10)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.plot(0, 0, "m.", markersize=10)
ax.set_title("Mean Elevation (m)")

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].pcolormesh(xi, yi, zg_frame, cmap="terrain", vmin=-10, vmax=10)
axs[0].set_title(f"frame_num = {FRAME_NUM}\nInstantaneous Elevation (m)")

axs[1].pcolormesh(
    xi,
    yi,
    zg_frame - zg_mean,
    cmap="jet",
    vmin=0,
    vmax=DZ_CMAP_SCALE,
)
axs[1].set_title("Instantaneous Elevation - Mean Elevation (m)")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.plot(0, 0, "m.", markersize=10)
    ax.label_outer()

# %
pc_laser_id = pc[pc.laser_id == 14]

xi = np.arange(-50, 0, 0.25)
frame_nums = np.arange(0, np.max(pc_laser_id.frame_num) + 1)


def low_fun(x):
    return np.percentile(x, 1)


mean_transect_z = bin_grid_1d(
    pc_laser_id.transformed_frame.x_meters,
    pc_laser_id.transformed_frame.z_meters,
    xi,
    low_fun,
)

z_grid_transect = bin_grid_2d(
    pc_laser_id.transformed_frame.x_meters,
    pc_laser_id.frame_num,
    pc_laser_id.transformed_frame.z_meters,
    xi,
    np.arange(0, np.max(pc_laser_id.frame_num) + 1),
    "mean",
)

height_above_min = z_grid_transect - mean_transect_z

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = ax.pcolormesh(
    frame_nums,
    xi,
    height_above_min.T,
    vmin=0,
    vmax=DZ_CMAP_SCALE,
)
ax.set_xlabel("Frame Number")
ax.set_ylabel("X (m)")
ax.set_title("Height Above Minimum (m)")
fig.colorbar(im, ax=ax, label="Height Above Minimum Transect(m)")
