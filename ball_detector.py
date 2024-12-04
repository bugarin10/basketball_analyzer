from ultralytics import YOLO
import numpy as np


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
