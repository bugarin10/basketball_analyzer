from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




def extract_landmarks(pose_landmarks, visibility=True):
    """
    Extracts x, y, and either visibility or presence from pose landmarks.

    Args:
        pose_landmarks: A collection of Mediapipe pose landmarks. In our case detection_result.pose_world_landmarks[0]
        visibility (bool): If True, includes visibility; otherwise, includes presence.

    Returns:
        list of lists: Each sublist contains [x, y, visibility/presence].
    """
    landmarks_list = []

    for landmark in pose_landmarks:
        if visibility:
            landmarks_list.append([landmark.x, landmark.y, landmark.visibility])
        else:
            landmarks_list.append([landmark.x, landmark.y, landmark.presence])

    return landmarks_list


def pose_estimation(path_to_image='image.jpg',visibility=True):

    """
        This functions return x,y and the visibility from an image

    """

    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(path_to_image)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    return extract_landmarks(detection_result.pose_world_landmarks[0], visibility=True)


if __name__=="__main__":
    print(f'keypoints for default image.jpg:\n{pose_estimation()}')