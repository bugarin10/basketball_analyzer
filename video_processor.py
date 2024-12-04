from ultralytics import YOLO
import numpy as np
import cv2
import os
from pose_estimation import pose_calc
from video_handler import VideoHandler


class VideoProcessor:
    def __init__(self):
        self.kp_model = YOLO("yolov8m.pt")
        self.video_handler = VideoHandler()
        self.videos = self.video_handler.get_unprocessed_video_paths()
        self.video_paths = [os.path.join(self.unprocessed_directory, vid) for vid in vids]

    def video_paths(self):
        """Create list of video paths from the unprocessed video directory"""

    
    def detect_ball(self, frame):
        result = self.kp_model.predict(frame, conf=0.5)

        ball_position = np.arange(2)[[i[5] == 32 for i in result[0].boxes.data.tolist()]][0]

        x_min, y_min, x_max, y_max, confidence, class_id = result[0].boxes.data.tolist()[
            ball_position
        ]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        return np.array([center_x, center_y, confidence])        

    def process_videos(self, video_path):
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
                kp = merge_keypoints(body_loc, bask_loc)
                keypoints.append(kp)

                processed_count += 1
                print(f"Processing frame {frame_count + 1}...")

            frame_count += 1

        print(keypoints[0])
        
        stabilized_kpts = head_stabilization(keypoints)

        print(stabilized_kpts[0])
        
        return stabilized_kpts