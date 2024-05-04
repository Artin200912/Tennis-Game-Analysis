# YOLO object detection model from ultralytics library
from ultralytics import YOLO
# OpenCV library for image processing
import cv2
# Module for serializing Python objects
import pickle
# Custom utility functions for bounding box operations
from utils.bbox_utils import *


class PlayerTracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path) # Initializes the PlayerTracker object with a YOLO Model loaded from the given path

    # This function takes the court_keypoints and player_detections as input and choose and filters 2 platyers
    def choose_and_filter_players(self, court_keypoints, player_detections):
        # This line extracts the first frame of the player_detection list of dictionary so this is basically taking the players detected in the first frame of the video
        player_detections_first_frame = player_detections[0] 
        # this line calls the choose_players method to select the players based on their distance to each court keypoint
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        # Crates an empty list for storing the filtered_players_detections
        filtered_player_detections = []
        # Iterating over each player_detection in the player_detections list of dicts
        for player_dict in player_detections:
            # it chooses the track_id and the bounding box corresponding to that id if the track_id is present in the choosen players dict
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            # then it addes the filtered_player_dict to the filtered_player_detections list of dict.
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending order to the most small minimum distances to be shown first in the list 
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks based on : first tuple, first element, which is going to give us the first player id, and the second tuple, first element is also going to give us the second_player_id
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    # This function is used to detect players in a video, it takes a list of frames and 2 optionals args that are going to determine that we will use the saved predictions or not

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # creates an empty list that is going to be a list of dictionary and each dict will contain the prediction for a specefic frame in that dict. so each dict would be a dict with keys as id and values as bounding box
        player_detections = []
        
        #This condition checks if both read_from_stub is True and stub_path is not None. If this condition is satisfied, it means the code should attempt to read player detections from a pickle file.
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
       # If the condition is not satisfied, indicating that player detections should be performed on the input frames:
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        #  This condition checks if stub_path is not None, indicating that player detections should be saved to a pickle file.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    # This method is used to detect players in a single frame.
    # It takes a single frame as input.
    def detect_frame(self, frame):
        # Perform object tracking on the given frame using the YOLO model.
        # The [0] is used to access the bounding box predictions from the tracked results.
        # The 'persist=True' parameter ensures that the model tracks objects across frames.
        results = self.model.track(frame, persist=True)[0]
        
        # Dictionary mapping class IDs to their names.
        id_name_dict = results.names

        # Dictionary to store detected player IDs and their bounding box coordinates.
        player_dict = {}

        # Iterate through all the bounding boxes detected by the model in the frame.
        for box in results.boxes:
            # Extract the track ID of the detected object and convert it to an integer.
            track_id = int(box.id.tolist()[0])
            
            # Extract the coordinates of the bounding box as a list.
            result = box.xyxy.tolist()[0]
            
            # Extract the class ID of the detected object.
            object_cls_id = box.cls.tolist()[0]
            
            # Map the class ID to the corresponding class name using the id_name_dict.
            object_cls_name = id_name_dict[object_cls_id]
            
            # If the detected object is classified as a person, store its bounding box coordinates.
            if object_cls_name == "person":
                player_dict[track_id] = result

        # Return the dictionary containing the detected players' bounding box coordinates.
        return player_dict

        
        return player_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = [] #initilizes an empty list which will going to be a list of frames but with drawn bounding boxes
        for frame, player_dict in zip(video_frames, player_detections): # we iterate over each frame and its corresponding bounding box by iterating in the zip of the video frames and the ball_detections list which is a list of dict where each dict is for a frame and its corresponding bounding box.
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():  # it iterates over the track_id which is 1 in this case, and its corresponding bounding box
                x1, y1, x2, y2 = bbox # then we draw them and save the new frames in the new list. 
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (225, 222, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (225, 225, 0), 2)
            output_video_frames.append(frame)
        
        return output_video_frames