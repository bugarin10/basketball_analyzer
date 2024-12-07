import numpy as np
import pandas as pd
import os

def find_origin(data_kp_path, origin_kp_path):
    keypoints = np.load(data_kp_path)
    print(keypoints.shape)
    df = pd.DataFrame(keypoints[0])
    print(df)
    print(f"right hip:{keypoints[0][23][:2]}")
    print(f"left hip:{keypoints[0][24][:2]}")
    right_hip_x = keypoints[0][23][0]
    right_hip_y = keypoints[0][23][1]
    left_hip_x = keypoints[0][24][0]
    left_hip_y = keypoints[0][24][1]

    origin_x = (right_hip_x + left_hip_x)/2
    origin_y = (right_hip_y + left_hip_y)/2
    origin = np.array([origin_x, origin_y])
    print(origin)
    np.save(origin_kp_path, origin)


if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.abspath(__file__))
    data_kp_path = os.path.join(root_directory, 'SA-Make-1.npy')
    origin_kp_path = os.path.join(root_directory, 'origin.npy')
    find_origin(data_kp_path, origin_kp_path)