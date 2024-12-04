from ultralytics import YOLO
import numpy as np
import cv2
import os
from pose_estimation import pose_calc
from file_handler import FileHandler


class VideoProcessor:
    def __init__(self):
        self.kp_model = YOLO("yolov8m.pt") # Set up basketball detection model
        self.video_handler = FileHandler() # Set up video handler (moving videos, video paths, etc)
        self.videos, self.unprocessed_directory = self.video_handler.get_unprocessed_video_paths() # List of videos in "data/unprocessed" folder
    
    def execute(self):
        if self.videos:
            self.video_paths = [os.path.join(self.unprocessed_directory, vid) for vid in self.videos]
            for video_name in self.videos:
                try:
                    video_path = os.path.join(self.unprocessed_directory, video_name)
                    kpts = self.process_video(video_path=video_path)
                    self.video_handler.save_keypoints(keypoint_data = kpts, file_name = video_name)
                except:
                    raise ValueError(f"Unable to process video: {video_name}")
        else:
            raise ValueError(f"No unprocessed videos in {self.unprocessed_directory}.")

    def detect_ball(self, frame):
        result = self.kp_model.predict(frame, conf=0.5)

        ball_position = np.arange(2)[[i[5] == 32 for i in result[0].boxes.data.tolist()]][0]

        x_min, y_min, x_max, y_max, confidence, class_id = result[0].boxes.data.tolist()[
            ball_position
        ]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        return np.array([center_x, center_y, confidence])        

    def process_video(self, video_path):
        # Open Video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return None
        
        # Calculate frame sampling frequency
        f_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        freq = max(1, f_total // 10)

        # Initialize counting parameters
        frame_count = 0
        processed_count = 0

        # Process Video
        keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_count % freq == 0:

                ######## Pose Estimation #########
                body_loc = pose_calc(frame)

                ######## Basketball Detection #########
                bask_loc = self.detect_ball(frame)

                # Merge keypoints
                kp = self.merge_keypoints(body_loc, bask_loc)
                keypoints.append(kp)

                processed_count += 1

            frame_count += 1
        
        stabilized_kpts = self.head_stabilization(keypoints)
        
        return stabilized_kpts
    
    def merge_keypoints(body_loc, bask_loc):
        """ This function merges the OpenPose Body Keypoints with the Basketball Location into one array"""
        # print(type(body_loc))
        # print(type(bask_loc))
        return np.concatenate((body_loc, bask_loc), axis=0)
    
    def head_stabilization(kp, target_head_location=(500, 350)): 
        """ This function centers the starting location of the head in this same frame location for every video"""

        head_x, head_y, _ = kp[0][0]  # REPLACE WITH HEAD KP FROM FIRST FRAME
        dx = target_head_location[0] - head_x
        dy = target_head_location[1] - head_y

        # Process all frames
        stabilized_keypoints = []
        for frame_index, frame_keypoints in enumerate(kp):
            # Verify only one person in the current frame
            # if frame_keypoints.shape[0] != 1:
            #     raise ValueError(f"Expected exactly one person in frame {frame_index + 1}, but found {frame_keypoints.shape[0]}.")

            # Stabilize keypoints for the single person in this frame
            frame_stabilized = frame_keypoints.copy()
            for keypoint_idx in range(frame_keypoints.shape[0]): 
                x, y, confidence = frame_keypoints[keypoint_idx]
                frame_stabilized[keypoint_idx] = [x + dx, y + dy, confidence]

            stabilized_keypoints.append(frame_stabilized)

        return stabilized_keypoints
            