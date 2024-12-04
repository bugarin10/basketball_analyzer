from ultralytics import YOLO
import numpy as np
import cv2


def detect_ball(frame):
    model = YOLO("yolov8m.pt")

    result = model.predict(frame, conf=0.5)

    ball_position = np.arange(2)[[i[5] == 32 for i in result[0].boxes.data.tolist()]][0]

    x_min, y_min, x_max, y_max, confidence, class_id = result[0].boxes.data.tolist()[
        ball_position
    ]
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)

    return np.array([center_x, center_y, confidence])

if __name__ == "__main__":
    video_path = 'data/01_videos/unprocessed/SA-Make-1.mov'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ball_loc = detect_ball(frame)
        print(ball_loc)
        break
