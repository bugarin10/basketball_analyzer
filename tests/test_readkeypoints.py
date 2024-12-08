import numpy as np
import pandas as pd
import os

def read_keypoints(kp_path):
    keypoints = np.load(kp_path)
    print(keypoints.shape)


if __name__ == "__main__":
    file_name = 'IMG_2662'
    root_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(root_directory, '..', '..'))
    keypoints_directory = os.path.join(parent_directory, 'data', '02_keypoints')
    keypoints_path = os.path.join(keypoints_directory, f'{file_name}.npy')
    #keypoints = np.load(keypoints_path)
    read_keypoints(keypoints_path)
