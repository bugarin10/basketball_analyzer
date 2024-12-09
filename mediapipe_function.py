from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseEstimator():
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='basketball_analyzer/pose_landmarker.task')
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options, 
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(self.options)


    def extract_landmarks(self, pose_landmarks, visibility=True):
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


    def pose_estimation(self, frame, visibility=True):

        """
            This functions return x,y and the visibility from an image

        """

        # STEP 3: Load the input image.
        #image = mp.Image.create_from_file(path_to_image)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = self.detector.detect(image)

        #print(detection_result)

        max_retries = 3
        for i in range(max_retries):
            try:
                detection_result = self.detector.detect(image)
                pose_data = self.extract_landmarks(detection_result.pose_landmarks[0], visibility=True)
                return pose_data
            except:
                print(f"POSE LANDMARK EXTRACTION FAILED: ATTEMPT {i}")
                continue
            
        print(f"FAILED TO EXTRACT MEDIAPIPE LANDMARKS")
        return None


if __name__=="__main__":

    import cv2

    video_path = 'data/01_videos/unprocessed/SA-Make-1.mov'
    pe = PoseEstimator()
    specific_frame_number = 0

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()

    # Set the frame position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, specific_frame_number)

    # Read the specific frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {specific_frame_number}.")
    else:
        # Process the specific frame
        print(f'keypoints for example frame:\n{pe.pose_estimation(frame=frame)}')