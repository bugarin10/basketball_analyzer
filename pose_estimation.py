import numpy as np

def pose_calc(frame):
    """ 
    Dummy Function for pose estimation
    Provided in COCO format
    """
    keypoints = np.array([
    [512, 328, 0.95],  # Nose
    [500, 320, 0.90],  # Left Eye
    [524, 320, 0.92],  # Right Eye
    [492, 340, 0.89],  # Left Ear
    [540, 340, 0.87],  # Right Ear
    [500, 400, 0.98],  # Left Shoulder
    [540, 400, 0.96],  # Right Shoulder
    [480, 500, 0.88],  # Left Elbow
    [560, 500, 0.92],  # Right Elbow
    [460, 600, 0.86],  # Left Wrist
    [580, 600, 0.89],  # Right Wrist
    [500, 600, 0.97],  # Left Hip
    [540, 600, 0.94],  # Right Hip
    [480, 700, 0.85],  # Left Knee
    [560, 700, 0.88],  # Right Knee
    [460, 800, 0.82],  # Left Ankle
    [580, 800, 0.84],  # Right Ankle
    ])
    return keypoints