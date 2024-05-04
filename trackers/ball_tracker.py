# Ultralytics for YOLO object detection
from ultralytics import YOLO
#Cv2 for openCV operations
import cv2
# Pickle for serializing  python objects 
import pickle
# pandas for data manipulations
import pandas as pd

# This class involves the functionalities related to ball detection task
class BallTracker:
    def __init__(self, model_path) -> None:
        # Initializes the BallTracker object with a YOLO Model loaded from the given path
        self.model = YOLO(model_path)

    # This functions takes self, ball_positions, self refers to the instance of the class and ball_positions refers ti the positions of the ball wich is a list of the positions of the ball. This function is used to fill out the outliers so we will not have any outliers anymore.
    def interpolate_ball_positions(self, ball_positions):
        # it iteraters over each item x in the ball_positions dictionary and searchs for '1'key, that refers to the ball_id, that is one because we only had 1 ball, so if the key 1 is available, we will return the value of that key which is a list containing the information about the ball's position
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # here, we are basically creating a pandas dictionary of the list extracted in the line above and the columns, are x1, y1, x2, y2 that are refering to the bounding box positions around the ball
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # This line fills out the outlier values in between gaps, by looking at the before and the after values
        df_ball_positions = df_ball_positions.interpolate()
        # This line fills out the remaining missing or NAN  values by repeating the last known value
        df_ball_positions = df_ball_positions.bfill()

       # This line converts the data frame back to a dictionary again by converting the dataframe into a list of dictionaries wehre each dict represnets a ball position
       # it does so by iterating over each row in the DataFrame, converting it to a dict with the ball_id = 1 and the key is the position_data : x 
       # x is a sublist representing the ball position cordinates fir each frame
       # Each sublist is generated from converting DataFrame values to a numpy array and then to a list
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
 
    def get_ball_shot_frames(self,ball_positions):
        # Method to identify frames where the tennis ball is hit
        # Takes a list of ball positions as input
        # Extracts ball positions from each item in the input list
        # Creates a DataFrame from the extracted ball positions with columns ['x1', 'y1', 'x2', 'y2']
        # Initializes a new column 'ball_hit' in the DataFrame with default value 0
        # Calculates the y-coordinate of the midpoint of each ball position and adds it to the DataFrame as 'mid_y'
        # Calculates a rolling mean of 'mid_y' with a window size of 5 to smooth out noise
        # Calculates the difference ('delta_y') between consecutive values of the rolling mean
        # Iterates over the DataFrame, checking for significant changes in 'delta_y' indicating a ball hit
        # If a significant change is detected, it checks for consecutive frames with similar changes to confirm the hit
        # Marks the frame as a ball hit if the consecutive change count exceeds a threshold
        # Returns a list of frame numbers where ball hits were detected

        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        # Method to detect tennis balls in each frame of the input video
        # Takes a list of frames as input
        # Optional parameters:
        #    - read_from_stub: If True and stub_path is provided, load pre-detected ball positions from a pickle file
        #    - stub_path: Path to the pickle file containing pre-detected ball positions
        # Returns a list of dictionaries, where each dictionary contains detected ball positions for a frame
        # If read_from_stub is True and stub_path is provided, loads pre-detected ball positions and returns them
        # Otherwise, iterates over each frame in the input list and detects ball positions using detect_frame method
        # Appends the detected ball positions to ball_detections list
        # If stub_path is provided, saves the detected ball positions to a pickle file
        # Returns the list of dictionaries containing detected ball positions for each frame

        # defines an empty list
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame) #detecting ball positions for each frame
            ball_detections.append(ball_dict) #adding that frame predictions to the ball_detections empty list
        
        if stub_path is not None:  #saves the ball_detections to a file if the sub_path is provided
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        # Method to detect tennis balls in a single frame using the YOLO model
        # Takes a single frame as input
        # Uses the YOLO model to predict ball positions in the input frame with a confidence threshold of 0.10
        # Returns a dictionary containing detected ball positions, where the key is the ball ID and the value is a list of bounding box coordinates
        # Each bounding box coordinate is represented as [x_min, y_min, x_max, y_max]

        results = self.model.predict(frame, conf=0.10)[0]

        ball_dict = {}
        for box in results.boxes: #We loop through all the bounding boxes predicted by the model
            result = box.xyxy.tolist()[0] # This is a list of lists, each inner list represents the cordinates of the bounding box. e.g [x_min, y_min, x_max, y_max] so by taking the element zero of the lists, we are basically taking the bounding box cordinates of the first box
            ball_dict[1] = result # we asign the ball_dict  key "1" to store the data of this particular prediction

        
        return ball_dict

    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = [] #initilizes an empty list which will going to be a list of frames but with drawn bounding boxes
        for frame, ball_dict in zip(video_frames, ball_detections): # we iterate over each frame and its corresponding bounding box by iterating in the zip of the video frames and the ball_detections list which is a list of dict where each dict is for a frame and its corresponding bounding box.
            # it iterates over the track_id which is 1 in this case, and its corresponding bounding box
            for track_id, bbox in ball_dict.items(): # then we draw them and save the new frames in the new list. 
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames