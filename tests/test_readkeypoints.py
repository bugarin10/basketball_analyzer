import numpy as np
import pandas as pd

def read_keypoints(kp_path):
    keypoints = np.load(kp_path)
    print(keypoints.shape)


if __name__ == "__main__":
    read_keypoints("data/02_keypoints/9_make.npy")