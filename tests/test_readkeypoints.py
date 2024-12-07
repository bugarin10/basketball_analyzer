import numpy as np
import pandas as pd

def read_keypoints(kp_path):
    keypoints = np.load(kp_path)
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

    print(origin_x, origin_y)


if __name__ == "__main__":
    read_keypoints("data/02_keypoints/SA-Make-1.npy")