# Velodyne HDL32e Operation
The Velodyne HDL32e is a 32 channel spinning lidar scanner operating at 903nm (or 905nm, depending on which manual you read). 

## Connecting to the sensor
The sensor communicates over ethernet, which is easily debugged using the software "Wireshark".

Plug the sensor into the computer, and open wireshark. Double click on the connection name corresponding to the port the lidar is connected to.  eg. ens0, etc. If unsure, click the one with a line chart next to it that shows significant data traffic. 

Once connected, you should see frequent UDP traffic coming from a fixed IP: eg. 192.168.1.201. This is the IP of the scanner.  In order to communicate with the scanner, your computer must be on 192.168.1.xxx.  Where xxx is a value other than the scanner. eg. 10

*If the scanner is set to have a destination IP address, you must set your computer to that exact value. eg. 1

Once your computer ip is correct, you can go to the browser and type the scanner IP address, and you'll see all the settings.

## Sensor Settings
In the browser, you can change the scanner ip, destination ip, inspect time sync, set phase angle, etc.

## Data
When the scanner is powered on, it will automatically start sending UDP packets over ethernet.  This data can be visualized in wireshark and is often saved as a pcap file. Each UDP transmission is encapsulated in multiple headers into a packet. These heads are used to direct the traffic over the network. eg. source ip, destination ip, packet length, source/destination port, etc. 

### Saving PCAP files
You can easily save a pcap file in wireshark or using tcpdump.  This contains each pcap packet and a computer timestamp for when the packet was received by the computer.  If your computer time and lidar time are synchronized, the difference in time between the time written to the UDP packet and the time it was received is often analyzed as latency.  A fixed latency value is expected due to processing of the lidar data, but spikes in latency can sometimes indicate an issue with the network bandwidth.

### Reading PCAP files in python
There are a few libraries to read pcap files (scapy, dpkt), but sometimes they can be a bit slow. The library "pytars" was developped to not be as feature rich, but is faster at reaching pcap packets.

ex.
```python 
PCAP_FILTERS = PcapPacketFilters(
    source_ip_addr="192.168.53.201",
    relative_time_gate_seconds=(0.0, 1),
)

PCAP_FILE = "foo.pcap"

with PcapReader(PCAP_FILE, packet_filters=pcap_filters) as pcap:
    for packet in pcap:
        udp_data_bytes = packet.udp_data.data
        # do something
        # * packet class also has packet header data
```

### Parsing UDP data
The udp data is all in the form of raw bytes. Raw bytes can be converted in python using struct.unpack, but using np.frombuffer with a custom dtype is often easier to follow.  eg.

```python
import numpy as np

example_data_1 = np.array([1, 2, 3], dtype=np.uint8)
example_data_2 = np.array([4, 5, 6], dtype=np.uint16)

data_as_bytes = bytes(example_data_1) + bytes(example_data_2)
# b'\x01\x02\x03\x04\x00\x05\x00\x06\x00'

dtype = np.dtype([("example_data_1", np.uint8, 3), ("example_data_2", np.uint16, 3)])
parsed_data = np.frombuffer(data_as_bytes, dtype=dtype)

print(parsed_data['example_data_1'][0])
# [1 2 3]
print(parsed_data['example_data_2'][0])
# [4 5 6]
```

## HDL32e data format
Each data packet has 1206 bytes and is sent to port 2368 by default. Each packet contains 12 blocks of 32 firings of laser data. eg. each laser fired 12 times. The documentation is inconsistent (possibly due to fw), but generally the packet structure contains:

- 12 x blocks
    - 2 byte: laser block id (constant)
    - 2 byte: rotational azimuth in hundeths of a degree (0-35999)
    - 32 x firing data (organized by known laser_id pattern)
        - 2 byte: distance information in 2mm increments (0 = no return)
        - 1 byte: intensity (0-255) where >100 = retro reflector
- 4 byte: gps timestamp (microseconds from top of hour for first firing in packet)
- 2 byte: factory information (constant)

### HDL32e telemetry format
The scanner also sends telemetry data with 512 bytes of data at ~200Hz. This includes the accelerometer data, gyroscope data, and GPS NMEA string if it's sync'd.

The details of the orientation of each accelerometer and the full packet format can be found in the documentation.

### Raw Data Timestamp 
Note that the raw pointcloud data does not have absolute time anywhere in the packet. It has relative time to the previous hour in microseconds.  In order to convert to absolute time, you need to read the telemetry data and parse the NMEA string to get the YYYYMMDD:hh value required to convert to absolute time.  This value then needs to be interpolated to every point cloud data packet, and edge cases need to be handled (eg. hour rollover).  

In the parser here, we just assume the pcap file isn't huge, and that the timestamp is within 1 hour of the median timestamp of the pcap file. There's probably better logic, but it will be ok if you keep pcap files small.

If there is no GPS data, the UTC time is nan and the timestamp is UTC, so it reverts back to 1970.  Then if it rolls over, it goes back to microseconds relative to 1970 again and it gets weird. Given this annoying packet structure, it really helps to have GPS data.

### Raw data Timestamp per point
In order to avoid crosstalk between laser firings, each firing is staggered in time. Rather than sending that same timestamp offset with every packet (and therefore growing the udp packet size), the equation for the timestamp of each firing in the 12 blocks is given and can be infered easily. See manual page 24.

## Raw Data -> Cartesian
Now that we have the time, motor azimuth, and laser_id (eg. which transceiver the return was from infered from the packet structure) - we need to convert to cartesian coordinates.  The azimuth and elevation angle of each laser is calibrated at the factory.  These values define the actual azimuth-elevation angles and how they differ from the design values due to lens distortion and manufacturing tolerances. In lieu of these calibration values, the default values can be used - though the data won't be as accurate.

#### Elevation angle per point
This value is read from the calibration file and is constant for each laser_id

#### Azimuth angle per point 
The azimuth value is computed based on the motor azimuth for each block, the relative timestamp of each firing relative to the timestamp of the first firing (which the motor azimuth corresponds to), and the motor RPM.  The motor rpm is inferred by linearly interpolating between each of the 12x blocks (extrapolating for block 12).

#### Range 
The range is directly reported in units of mm.

#### Dual returns
If dual return mode is on, the 32 blocks only reflect 6 true blocks. The first block is one return, the second block is another return (if applicable).  If the second block == first block range, then no second return value is present.  Refer to the manual on what each return means (eg. strongest, last, etc).  Two metrics are computed based on this dual return info:
- num_returns: number of total returns in that point's firing
- return_num: the return number that this point came from

So for example. return_num 1 of 2 num_returns means this point is the first return of two returns.

#### Frame num
Each time the motor azimuth crosses 0 degrees, it increments the frame count. This is helpful for visualizing each "frame" / "sweep".  

## Using the read_hdl32_pcap_pointcloud class

```python 

# constants to read the pcap file
PCAP_FILE = r"foo.pcap"

PCAP_FILTERS = PcapPacketFilters(
    relative_time_gate_seconds=(0.5, 5),
)

pc = read_hdl32e_pcap_pointcloud(
    pcap_file=PCAP_FILE,
    pcap_filters=PCAP_FILTERS,
    calibration_file=None,
    extrinsic_4x4_to_transformed_from_sensor=None,
    include_null_returns=False,
    include_telemetry=True,
    name="test",
)

# example plot the data
fig,ax = plt.subplots(1,1)
ax.plot(pc.sensor_frame.x_meters, pc.sensor_frame.y_meters,'.')

```