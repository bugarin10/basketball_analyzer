import os
import shutil
import numpy as np

class FileHandler:
    def __init__(self):
        # Set up the unprocessed and processed directories relative to the repository
        self.root_directory = os.path.dirname(os.path.abspath(__file__))
        self.unprocessed_directory = os.path.join(self.root_directory, 'data', '01_videos', 'unprocessed')
        self.processed_directory = os.path.join(self.root_directory, 'data', '01_videos', 'processed')
        self.keypoints_directory = os.path.join(self.root_directory, 'data', '02_keypoiunts')

    def get_unprocessed_video_paths(self):
        files = os.listdir(self.unprocessed_directory)
        vids = [file for file in files if file.endswith('.mov')]
        #vids = [os.path.join(self.unprocessed_directory, vid) for vid in vids]
        if len(vids) > 0:
            return vids, self.unprocessed_directory
        else: return None

    def move_video(self, video_name):
        prev_path = os.path.join(self.unprocessed_directory, video_name)
        new_path = os.path.join(self.processed_directory, video_name)
        shutil.move(prev_path, new_path)
        print(f"Moved {video_name} to {self.processed_directory}")
    
    def save_keypoints(self, keypoint_data:np.array, file_name:str):
        # Define the file path for saving the matrix
        file_path = os.path.join(self.keypoints_directory, f"{file_name}.npy")
        if not os.path.isfile(file_path):
            np.save(file_path, keypoint_data)
            print(f"{file_name}.npy saved.")
        else:
            print(f"{file_name}.npy already exists in data/02_keypoints directory")


if __name__ == "__main__":
    handler = FileHandler()
    vids = handler.get_unprocessed_video_paths()
    for vid in vids:
        handler.move_video(vid)
    