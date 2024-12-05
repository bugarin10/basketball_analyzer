from ultralytics import YOLO
import numpy as np
import cv2


class BallDetector():
    def __init__(self):

        self.model = YOLO("yolov8m.pt")

    def detect_ball(self, frame):

        result = self.model.predict(frame, conf=0.1)

        detections = [i[5] == 32 for i in result[0].boxes.data.tolist()]

        ball_position = np.arange(len(detections))[detections][0]

        
        x_min, y_min, x_max, y_max, confidence, class_id = result[0].boxes.data.tolist()[
            ball_position
        ]
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        return np.array([[center_x, center_y, confidence]]) 
      
    def plot_ball_on_frame(self, frame, ball_data):
        """
        Plots the ball's location on the frame.
        
        Parameters:
        - frame: The video frame (cv2 image) to draw on.
        - ball_data: A NumPy array with the format [[center_x, center_y, confidence]].
        """
        if ball_data is None:
            print("No ball detected to plot.")
            return frame  # Return the frame unchanged
        
        # Extract ball data
        center_x, center_y, confidence = ball_data[0]
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


if __name__ == "__main__":
    video_path = 'data/01_videos/unprocessed/SA-Make-1.mov'
    specific_frame_number = 27  # Replace this with the frame number you want to read

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
        ball_detector = BallDetector()
        ball_loc = ball_detector.detect_ball(frame)
        print(f"Ball location at frame {specific_frame_number}: {ball_loc}")
        ball_detector.plot_ball_on_frame(frame, ball_data = ball_loc)

    # Release the video capture object
    cap.release()
