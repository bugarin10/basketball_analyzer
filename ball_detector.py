from ultralytics import YOLO
import numpy as np
import cv2
import os


class BallDetector():
    def __init__(self, frame_shape = None):

        self.model = YOLO("yolov8x.pt")
        self.frame_shape = frame_shape
        self.conf_thresh = 0.08

    def detect_ball(self, frame, verbose=False):

        result = self.model.predict(frame, conf=self.conf_thresh, verbose=verbose)

        detections = [i[5] == 32 for i in result[0].boxes.data.tolist()]

        if sum(detections) == 0:
            return None

        ball_position = np.arange(len(detections))[detections][0]
        
        x_min, y_min, x_max, y_max, confidence, class_id = result[0].boxes.data.tolist()[
            ball_position
        ]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        loc = np.array([[center_x, center_y, confidence]])

        return self.normalize_coordinates(loc)
      
    def plot_ball_on_frame(self, frame, ball_data):
        """
        Plots the ball's location on the frame.
        
        Parameters:
        - frame: The video frame (cv2 image) to draw on.
        - ball_data: A NumPy array with the format [[center_x, center_y, confidence]].
        """
        if ball_data is None:
            cv2.imshow('Ball Detection', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("No ball detected to plot.")
            return frame  # Return the frame unchanged
        
        # Extract ball data
        center_x, center_y, confidence = ball_data[0]
        # Get frame dimensions
        frame_height, frame_width = self.frame_shape

        # Denormalize basketball coordinates for the first frame
        center_x = int(center_x * frame_width)
        center_y = int(center_y * frame_height)
        center = (int(center_x), int(center_y))
        center_text = (int(center_x + 15), int(center_y - 10))

        # Draw a circle at the ball's position
        cv2.circle(frame, center, 10, (0, 255, 0), -1)  # Green filled circle

        # Annotate confidence score
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f}",
            center_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.imshow('Ball Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def last_basketball_detection_binary(self, video_path):
        "Binary search for last frame with detected basketball"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return None

        left = 0
        right = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames
        last_detected_frame = -1
        count = 0
        while (left < right)&(count<20):
            #print(last_detected_frame)
            mid = ((left + right + 1) // 2)
            #print(left, right, mid)
            print(f"Binary Search For Last Visible Basketball. Processing Frame: {mid}", end='\r')
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {mid}. Stopping search.")
                break
            if last_detected_frame == -1:
                self.frame_shape = frame.shape[:2]

            ball_loc = self.detect_ball(frame)
            if ball_loc is not None:
                left = mid
                last_detected_frame = mid
            else:
                right =  mid - 1
            
            count += 1

        cap.release()

        if count == 20:
            print("LAST BASKETBALL FRAME DID NOT CONVERGE")
            return None
        
        cap.release()

        return last_detected_frame
    
    def last_basketball_detection(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            exit()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        while i < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {i}. Stopping search.")
                break
            if i == 0:
                self.frame_shape = frame.shape[:2]
            ball_loc = self.detect_ball(frame)
            if ball_loc is not None:
                i += 1
                continue
            else:
                break
        cap.release()
        return i-1

        
    def normalize_coordinates(self, loc):
        """Normalize location of basketball in the video"""
        x_loc, y_loc, conf = loc[0]
        h, w = self.frame_shape
        x_norm = x_loc / w
        y_norm = y_loc / h
        return np.array([[x_norm, y_norm, conf]])

if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(root_directory)
    unprocessed_directory = os.path.join(parent_directory, 'data', '01_videos', 'unprocessed')
    file_name = '87_miss'
    video_path = os.path.join(unprocessed_directory, f'{file_name}.mov')
    ball_detector = BallDetector()
    #specific_frame_number = ball_detector.last_basketball_detection(video_path=video_path)
    specific_frame_number = 167
    if specific_frame_number is None:
        exit
    #frames = numbers = list(range(86, 104))

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
        frame_shape = frame.shape[:2]
        ball_detector = BallDetector(frame_shape=frame_shape)
        ball_loc = ball_detector.detect_ball(frame, verbose=True)
        if ball_loc is None:
            exit
        print(f"Ball location at frame {specific_frame_number}: {ball_loc}")
        ball_detector.plot_ball_on_frame(frame, ball_data = ball_loc)

    # Release the video capture object
    cap.release()
