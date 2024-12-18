import os
import shutil
import numpy as np
import json

class FileHandler:
    def __init__(self):
        # Set up the unprocessed and processed directories relative to the repository
        self.root_directory = os.path.dirname(os.path.abspath(__file__))
        self.parent_directory = os.path.dirname(self.root_directory) #Used to access data directories outside of repository
        self.unprocessed_directory = os.path.join(self.parent_directory, 'data', '01_videos', 'unprocessed')
        self.processed_directory = os.path.join(self.parent_directory, 'data', '01_videos', 'processed')
        self.errored_directory = os.path.join(self.parent_directory, 'data', '01_videos', 'errored')
        self.keypoints_directory = os.path.join(self.parent_directory, 'data', '02_keypoints')
        self.origin_directory = os.path.join(self.root_directory, 'data', '00_origin_data')
        self.misc_directory = os.path.join(self.root_directory, 'data', '99_misc')

    def get_origin(self):
        origin_array = os.path.join(self.origin_directory, 'origin.npy')
        return np.load(origin_array)

    def get_unprocessed_video_paths(self):
        files = os.listdir(self.unprocessed_directory)
        vids = [file for file in files if file.lower().endswith('.mov')]
        #vids = [os.path.join(self.unprocessed_directory, vid) for vid in vids]
        if len(vids) > 0:
            return vids, self.unprocessed_directory
        else: return None

    def move_video(self, success:bool, video_name):
        prev_path = os.path.join(self.unprocessed_directory, video_name)
        if not success:
            new_path = os.path.join(self.errored_directory, video_name)
        else:
            new_path = os.path.join(self.processed_directory, video_name)
        shutil.move(prev_path, new_path)
        print(f"Moved {video_name} to {self.processed_directory}")
        return None
    
    def save_keypoints(self, keypoint_data:np.array, file_name:str):
        # Define the file path for saving the matrix
        if keypoint_data is None:
            print("KEYPOINT DATA DOES NOT EXIST. NOT SAVING")
            return False

        file_path = os.path.join(self.keypoints_directory, f"{file_name[:-4]}.npy")
        if not os.path.isfile(file_path):
            try:
                np.save(file_path, keypoint_data)
                print(f"{file_name}.npy saved.")
                return True
            except Exception as e:
                raise ValueError(f"Error saving file {file_name}, Error: {e}")
                
        else:
            print(f"{file_name}.npy already exists in data/02_keypoints directory")
            return True
        

    def save_final_frames(self, new_data):
        file_path = os.path.join(self.misc_directory, 'final_frames.json')
        
        # Check if the JSON file exists and load its content
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = {}  # Handle corrupted or empty files
        else:
            existing_data = {}

        # Update the existing data with the new data
        # Assuming `new_data` is a dictionary and you're merging it with `existing_data`
        existing_data.update(new_data)

        # Write the updated data back to the JSON file
        with open(file_path, "w") as file:
            json.dump(existing_data, file, indent=4)  # Use indent for readability

        print(f"JSON file updated and saved to {file_path}")



if __name__ == "__main__":
    # handler = FileHandler()
    # origin = handler.get_origin()
    # print(origin)
    root_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(root_directory)
    unprocessed_directory = os.path.join(parent_directory, 'data', '01_videos', 'unprocessed')
    print(unprocessed_directory)
    