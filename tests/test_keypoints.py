import numpy as np
np.set_printoptions(suppress=True)

def test_keypoints(kp_path):
    kp = np.load(kp_path)
    print(kp)
    print(kp.shape)

if __name__ == "__main__":
    test_keypoints("./data/02_keypoints/SA-Make-1.npy")