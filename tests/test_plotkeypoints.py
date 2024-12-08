import cv2
import numpy as np

def plot_keypoints_first_frame(frame, frame_kp, color_basketball=(0, 255, 0), color_pose=(255, 0, 0)):
    """
    Plots keypoints for the first frame and basketball coordinates on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to overlay the points on.
        keypoints (numpy.ndarray): Array of size (11, 34, 3) containing keypoints for multiple frames.
        basketball_coords (tuple): Normalized (x, y) coordinates of the basketball for the first frame.
        color_basketball (tuple): RGB color for the basketball point. Default is green.
        color_pose (tuple): RGB color for the pose landmarks. Default is blue.

    Returns:
        numpy.ndarray: The frame with the points plotted.
    """
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Denormalize basketball coordinates for the first frame
    basketball_x = int(frame_kp[33][0] * frame_width)
    basketball_y = int(frame_kp[33][1] * frame_height)

    # Draw basketball point
    cv2.circle(frame, (basketball_x, basketball_y), radius=10, color=color_basketball, thickness=-1)

    # Plot pose keypoints for the first frame
    for kp in frame_kp:
        pose_x = int(kp[0] * frame_width)  # Denormalize X
        pose_y = int(kp[1] * frame_height)  # Denormalize Y
        confidence = kp[2]  # Confidence value (not used for plotting here)

        # Only plot points with high confidence
        if confidence > 0.5:  # Threshold for visibility
            cv2.circle(frame, (pose_x, pose_y), radius=5, color=color_pose, thickness=-1)

    # Return the frame with points plotted
    return frame

if __name__ == "__main__":

    file_name = "9_make"
    video_path = f'data/01_videos/processed/{file_name}.mov'

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()

    # Set the frame position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 37)

    # Read the specific frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {37}.")
    

    # Example input: Keypoints array (11 frames, 34 keypoints, 3 values per keypoint)
    keypoints = np.load(f"data/02_keypoints/{file_name}.npy")
    frame_kp = keypoints[0] #### CHANGE BASED ON FRAME NUMBER

    # Plot keypoints and basketball location on the first frame
    output_frame = plot_keypoints_first_frame(frame, frame_kp)

    # Display the frame
    cv2.imshow("Keypoints and Basketball", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
