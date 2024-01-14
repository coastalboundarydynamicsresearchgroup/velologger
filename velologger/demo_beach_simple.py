# %%
import matplotlib.pyplot as plt
import numpy as np
from pytars.utils.custom_cmaps import get_fixed_cmap
from pytars.utils.gridding import bin_grid_1d, bin_grid_2d
from pytars.utils.timing import mean_datetime64

from velologger.hdl32e.read_hdl32e_pcap_pointcloud import read_hdl32e_pcap_simple

# list all the pcap files you want to read
PCAP_FILE_LIST = [
    r"/Users/rslocum/Downloads/A.pcap",
    r"/Users/rslocum/Downloads/B.pcap",
    r"/Users/rslocum/Downloads/C.pcap",
    r"/Users/rslocum/Downloads/D.pcap",
]

# Constants for visualization
DX = 0.25  # cross-shore and along-shore grid spacing
DZ_CMAP_SCALE = 0.5  # cmap for pcolor
FRAME_NUM = 3  # instantaneous framenum to plot

# read the pcap files (reading too many might take a ton of memory)
pc = read_hdl32e_pcap_simple(
    pcap_file=PCAP_FILE_LIST,
    name="sample collect",  # this is nice if you load multiple pointclouds so you can name them
)

# compute mean z_grid
xi = np.arange(-50, 40, DX)
yi = np.arange(-10, 20, DX)


# define a function that takes the lowest percentile, rather than the min
def low_fun(x):
    return np.percentile(x, 1)


# grid the data so each cell is the results of "low_fun"
#   on all values that fell within that cell
zg_mean = bin_grid_2d(
    pc.transformed_frame.x_meters,
    pc.transformed_frame.y_meters,
    pc.transformed_frame.z_meters,
    xi,
    yi,
    low_fun,
)

# make a new pc object for just the specific frame
pc_frame = pc[pc.frame_num == FRAME_NUM]

# take the mean of an instantaneous frame
zg_frame = bin_grid_2d(
    pc_frame.transformed_frame.x_meters,
    pc_frame.transformed_frame.y_meters,
    pc_frame.transformed_frame.z_meters,
    xi,
    yi,
    "mean",
)

# visualize the min z
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.pcolormesh(xi, yi, zg_mean, cmap="terrain", vmin=-10, vmax=10)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.plot(0, 0, "m.", markersize=10)
ax.set_title("Min (1 percentile) Elevation (m)")

# visualize the instantaneous z - mean z
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
axs[1].set_title("Instantaneous Elevation - Min (1%) Elevation (m)")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.plot(0, 0, "m.", markersize=10)
    ax.label_outer()

# create a new pc object for only laser_id == 14
# also only use frames after the first frame (since that one will be partial)
pc_laser_id = pc[(pc.laser_id == 14) & (pc.frame_num > 0)]

# define a new grid for the transect
xi = np.arange(-50, 0, 0.25)
# create a list of all the frame_nums
frame_nums = np.arange(1, np.max(pc_laser_id.frame_num) + 1)
# compute mean time for each frame
frame_nums_time = [
    mean_datetime64(pc_laser_id.datetime64[pc_laser_id.frame_num == frame_num])
    for frame_num in frame_nums
]

# grid 1d - take the low_fun of all z values in each cell
low_transect_z = bin_grid_1d(
    pc_laser_id.transformed_frame.x_meters,
    pc_laser_id.transformed_frame.z_meters,
    xi,
    low_fun,
)

# grid all transects, gridding by frame_num so it's a 2D timeseries
z_grid_transect = bin_grid_2d(
    pc_laser_id.transformed_frame.x_meters,
    pc_laser_id.frame_num,
    pc_laser_id.transformed_frame.z_meters,
    xi,
    frame_nums,
    "mean",
)
# compute the height above the minimum transect for easy visualization
height_above_min = z_grid_transect - low_transect_z

# Plot the height above minimum for each frame
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = ax.pcolormesh(
    frame_nums_time,
    xi,
    height_above_min.T,
    vmin=0,
    vmax=DZ_CMAP_SCALE,
)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("X (m)")
ax.set_title("Height Above Minimum (m)")
ax.invert_yaxis()
fig.colorbar(im, ax=ax, label="Height Above Minimum Transect(m)")
# % plot transects colored by time
CMAP_RANGE = (10, 45)
CMAP_STEP = 5
CMAP_NAME = "jet"
XLIM = (-40, -5)
YLIM = (-9.5, -4)
OBSERVED_BEACH_X_EXTENT = -31
all_frame_nums = np.arange(CMAP_RANGE[0], CMAP_RANGE[1], CMAP_STEP)
# create a fixed colormap rather than a legend
cmap, norm = get_fixed_cmap(all_frame_nums, CMAP_NAME)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# plot the low transect value as gray to represent the beach (assumes no excessive erosion)
low_transect_z_only_beach = low_transect_z.copy()
low_transect_z_only_beach[
    xi < OBSERVED_BEACH_X_EXTENT
] = np.nan  # -31 hardcoded, but could do algorithmically
ax.fill_between(
    xi, low_transect_z_only_beach, np.ones_like(low_transect_z) * -10, color="gray", alpha=0.5
)
# plot each transect
for i, frame_num in enumerate(all_frame_nums):
    ind = (pc_laser_id.frame_num == frame_num) & (pc_laser_id.transformed_frame.x_meters < 0)
    ax.plot(
        pc_laser_id.transformed_frame.x_meters[ind],
        pc_laser_id.transformed_frame.z_meters[ind],
        ".-",
        alpha=0.5,
        color=cmap(norm(frame_num)),  # color based on the cmap
    )

ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
# make colorbar
cbar = fig.colorbar(None, ax=ax, cmap=cmap, norm=norm)
cbar.set_ticks(all_frame_nums)
# label based on time, rather than frame num
tick_labels = [f"{str(frame_nums_time[frame_num])[11:-4]}" for frame_num in all_frame_nums]
cbar.set_ticklabels(tick_labels)

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Transect Colored by Frame Number")
