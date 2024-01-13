import numpy as np


def calc_roll_pitch(acc_x, acc_y, acc_z, is_degrees: bool = True):
    roll = np.arctan2(acc_y, acc_z)
    pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

    # Convert from radians to degrees
    if is_degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)

    return roll, pitch
