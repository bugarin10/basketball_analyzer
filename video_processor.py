from ultralytics import YOLO
import numpy as np
import cv2
import os
from mediapipe_function import PoseEstimator
from file_handler import FileHandler
from ball_detector import BallDetector


class VideoProcessor:
    def __init__(self):
        self.kp_model = YOLO("yolov8m.pt") # Set up basketball detection model
        self.file_handler = FileHandler() # Set up video handler (moving videos, video paths, etc)
        self.ball_detector = BallDetector() # Ball Detection Object w/ Yolo Model
        self.pose_estimator = PoseEstimator()
        self.videos, self.unprocessed_directory = self.file_handler.get_unprocessed_video_paths() # List of videos in "data/unprocessed" folder
        self._current_step = None

    def _identify_error_step(self):
        """Helper method to provide the current step for debugging."""
        if not hasattr(self, '_current_step'):
            return "Unknown"
        return self._current_step

    def execute(self):
        if self.videos:
            video_count = 0
            for video_name in self.videos:
                try:
                    self._current_step = "Constructing video path"
                    video_path = os.path.join(self.unprocessed_directory, video_name)

                    self._current_step = "Processing video"
                    kpts = self.process_video(video_path=video_path)

                    self._current_step = "Saving keypoints"
                    self.file_handler.save_keypoints(keypoint_data=kpts, file_name=video_name)

                    self._current_step = "Transferring video"
                    self.file_handler.move_video(video_name=video_name)

                    video_count+=1
                    
                except Exception as e:
                    raise ValueError(f"An error occurred while processing video: {video_name}. Step: {self._identify_error_step()}, Error: {e}")
        else:
            raise ValueError(f"No unprocessed videos in {self.unprocessed_directory}.") 

        print(f"Video Processing Complete. {video_count} video(s) processed.")    

    def process_video(self, video_path):
        # Open Video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return None
        
        # Calculate frame sampling frequency
        f_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        freq = max(1, f_total // 10)
        print(f"TOTAL FRAMES {f_total}")
        print(f"FREQUENCY {freq}")

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
            if (frame_count % freq == 0) & (frame_count < 20):
                print(f"FRAME NUMBER {frame_count}")
                ######## Pose Estimation #########
                body_loc = self.pose_estimator.pose_estimation(frame)

                ######## Basketball Detection #########
                bask_loc = self.ball_detector.detect_ball(frame)
                print(bask_loc)

                # Merge keypoints
                kp = self.merge_keypoints(body_loc, bask_loc)
                keypoints.append(kp)

                processed_count += 1

            frame_count += 1
        
        stabilized_kpts = self.head_stabilization(keypoints)
        
        return stabilized_kpts 
    
    def merge_keypoints(self, body_loc, bask_loc):
        """ This function merges the OpenPose Body Keypoints with the Basketball Location into one array"""
        # print(type(body_loc))
        # print(type(bask_loc))
        return np.concatenate((body_loc, bask_loc), axis=0)
    
    def head_stabilization(self, kp, target_head_location=(500, 350)): 
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
    
if __name__ == "__main__":
    vp = VideoProcessor()
    vp.execute()
            