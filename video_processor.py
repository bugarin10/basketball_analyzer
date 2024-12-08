from ultralytics import YOLO
import numpy as np
import cv2
import os
from .mediapipe_function import PoseEstimator
from .file_handler import FileHandler
from .ball_detector import BallDetector


class VideoProcessor:
    def __init__(self):
        self.file_handler = FileHandler() # Set up video handler (moving videos, video paths, etc)
        self.ball_detector = BallDetector() # Ball Detection Object w/ Yolo Model
        self.pose_estimator = PoseEstimator()
        self.videos, self.unprocessed_directory = self.file_handler.get_unprocessed_video_paths() # List of videos in "data/unprocessed" folder
        self._current_step = None
        self.origin = self.file_handler.get_origin()  # PREDETERMINED STARTING HIP LOCATION FOR EVERY VIDEO

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
                    success = self.file_handler.save_keypoints(keypoint_data=kpts, file_name=video_name)

                    self._current_step = "Transferring video"
                    _ = self.file_handler.move_video(success, video_name=video_name)

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
        desired_frames = 20
        # f_total = self.ball_detector.last_basketball_detection(video_path) ###BASKETBALL
        f_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if f_total is None:
            return None
        frame_indices = set(np.linspace(0, f_total - 1, num=desired_frames, dtype=int))
        if len(frame_indices) < desired_frames:
            print(f"LOW FRAMES DETECTED. VIDEO {video_path}")
            return None
        
        print(f"TOTAL FRAMES {f_total}")
        print(f"TRIMMED FRAME NUMBER TO: {desired_frames}")

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
            if frame_count in frame_indices:
                print(f"FRAME NUMBER {frame_count}")
                ######## Pose Estimation #########
                body_loc = np.array(self.pose_estimator.pose_estimation(frame)) # .reshape(1, -1)
                #print(body_loc.shape)

                ######## Basketball Detection #########
                #bask_loc = self.ball_detector.detect_ball(frame) ###BASKETBALL
                #print(bask_loc) 

                # Merge keypoints
                #kp = self.merge_keypoints(body_loc, bask_loc) ###BASKETBALL
                #keypoints.append(kp) ###BASKETBALL
                keypoints.append(body_loc)


                processed_count += 1
            
            frame_count += 1

        kp_array = np.array(keypoints)
        stabilized_kpts = self.head_stabilization(kp_array)
        
        return stabilized_kpts 
    
    def merge_keypoints(self, body_loc, bask_loc):
        """ This function merges the OpenPose Body Keypoints with the Basketball Location into one array"""
        # print(type(body_loc))
        # print(type(bask_loc))
        return np.concatenate((body_loc, bask_loc), axis=0)
    
    def head_stabilization(self, kp): 
        """ This function centers the starting location of the head in this same frame location for every video
        
        ***** MAY BE OBSOLETE *****
        
        Mediapipe pose estimator has coordinate system relative to body hips "world-coordinates". 
        Need method for converting ball location to world-coordinate system.
        """

        right_hip_x = kp[0][23][0]
        right_hip_y = kp[0][23][1]
        left_hip_x = kp[0][24][0]
        left_hip_y = kp[0][24][1]

        vid_origin_x = (right_hip_x + left_hip_x)/2
        vid_origin_y = (right_hip_y + left_hip_y)/2

        dx = self.origin[0] - vid_origin_x
        dy = self.origin[1] - vid_origin_y

        # Process all frames
        stabilized_keypoints = []
        for frame_index, frame_keypoints in enumerate(kp):
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
    # root_directory = os.path.dirname(os.path.abspath(__file__))
    # print(root_directory)
    # data_kp_path = os.path.join(root_directory, 'data','00_origin_data','SA-Make-1.npy')
    # print(data_kp_path)
    # kp = np.load(data_kp_path)
    # vp.head_stabilization(kp)
            