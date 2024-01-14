import numpy as np
from pytars.transforms.transform3d import create_rotation_matrix_4x4


def calc_roll_pitch(acc_x, acc_y, acc_z, is_degrees: bool = True):
    roll = np.arctan2(acc_y, acc_z)
    pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

    # Convert from radians to degrees
    if is_degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)

    return roll, pitch


def get_hdl32e_extrinsic_4x4(mean_accel_x, mean_accel_y, mean_accel_z):
    roll_rad, pitch_rad = calc_roll_pitch(
        mean_accel_x, mean_accel_y, mean_accel_z, is_degrees=False
    )
    return create_rotation_matrix_4x4(0, 0, 0, pitch_rad, -roll_rad, np.pi / 2, is_degrees=False)
