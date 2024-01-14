# velologger
Code to read HDL32e pcap files
## Overview
The HDL32e sends UDP point cloud packets and separate UDP telemetry packets. UDP traffic is often stored in the PCAP file format.
### UDP Point Cloud Packet
Raw data:
- timestamp (microseconds from the top of the last hour)
- 12 blocks of 32 x (range, reflectivity) data
- 12 azimuth values (1 x per block)

Processing:
- elevation angle per channel is loaded from a default calibration file
- timing is calculated relative to the timestamp based on an equation in the datasheet
- timing is converted to "utc time" by cross-referencing the telemetry data packets. This raw data in these packets has no absolute time! 
- azimuth angle is interpolated based on timing and motor rpm. RPM is estimated per packet by looking at the azimuth delta between sequential blocks.

### UDP Telemetry Packet
Raw data:
- timestamp (microseconds from the top of the last hour)
- GPRMC message in ascii format (when present in the data)
- acceleration values 2x per accel_x, accel_y, accel_z
- 2x gyroscope values 
- 3x temperature values

Processing:
- GPRMC message is parsed and processed to convert timestamps to UTC time

## Basic PCAP Reading
```python
from velologger.hdl32e.read_hdl32e_pcap_pointcloud import read_hdl32e_pcap_simple

# list all the pcap files you want to read
PCAP_FILE_LIST = [
    r"/Users/foo/A.pcap",
    r"/Users/foo/B.pcap",
    r"/Users/foo/C.pcap",
]

# read the pcap files (reading too many might take a ton of memory and crash)
pc = read_hdl32e_pcap_simple(
    pcap_file=PCAP_FILE_LIST,
    name="sample collect", # this is nice if you load multiple pointclouds so you can name them
)

# isolate only frame 10 (frame == full 360 deg sweep)
pc_one_frame = pc[pc.frame_num==10]

# scatter plot the data
fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.plot(pc_one_frame.transformed_frame.x_meters,
        pc_one_frame.transformed_frame.y_meters,
        '.',)
```

### Object variables
The pc object has the following variables:
``` python
    sensor_frame: Coordinates
    transformed_frame: Coordinates
    laser_id: np.ndarray # which laser_id the point is from (0-31)
    datetime64: np.ndarray # UTC timestamp in units of us
    reflectivity: np.ndarray
    frame_num: np.ndarray # increments every time the azimuth crosses 0
    return_num: np.ndarray # for multiple returns, indicates if which return num it was
    num_returns: np.ndarray # indicates the total number of returns when in dual return
    pcap_file_names: List[Path] 
    pcap_file_ind: np.ndarray
    name: str
```

### Coordinates 
The coordinates object holds the 3D data for each point.  
``` python
    x_meters: np.ndarray
    y_meters: np.ndarray
    z_meters: np.ndarray
    azimuth_degrees: np.ndarray
    elevation_degrees: np.ndarray
    range_meters: np.ndarray
    transform_4x4_to_current_from_original: np.ndarray # 4x4 matrix
    coordinate_system: CoordinateSystem # object defining xyz-sph convention
```
#### Sensor Frame
The sensor coordinate frame stored as "sensor_frame".  This is based on CAD conventions. 

#### Transformed Frame
This can be set manually by inputting a 4x4 transformation matrix into the reader. If it is not input, the mean acceleration values for all the data are used to estimate the roll and pitch.  The data and transformed using these values.

#### Why are these separate?
This makes it easy to visualize sensor data in both frames. eg. scatter plot sensor spherical coordinates colored by transformed z.  
## Create a virtual environment 
With python 3.8-3.10 installed on your computer - run the following
```
cd C:/path/to/folder/velologger
C:/full/path/to/python3.xx/python.exe -m venv velologger_venv
# if in vscode, say yes to automatically activate
# otherwise on windows:
z:/velologger/velologger_env/Scripts/Activate.ps1
# now install dependencies
pip install numpy, scipy, matplotlib, pytars, jupyter
```
## Troubleshooting
### velologger python path not set correctly
You'll get import errors when importing velologger.x

- make sure your `.env` file contains the correct PYTHONPATH to the main velologger folder.
- vscode will automatically use this. If you're not using vscode - you'll have to set the system environment variables.
- check that it's working by running python, then:
```python 
import sys
print(sys.path) # the output should have C:/path/to/folder/velologger
```

### dependencies not updated
Symptom: weird errors like functions don't exist or functionality doesn't work as expected. 

Sometimes you might not have the latest version of a library, check pypi to see if there's a more recent version. There's more robust ways to manage this like being explicity in requirements.txt or use poetry, but it can be fixed quickly.  Say for example, pytars is out of date.

** pytars is a repo that has some pcap reading functionality that Richie and Chase developed. It's very beta and sometimes has had some breaking changes.
```
pip install pytars --force-reinstall
```
After forcing the reinstall, run `pip list` from your venv and check to ensure it's the right version.

```
# alternate brute force approach
pip uninstall pytars
pip install pytars
```