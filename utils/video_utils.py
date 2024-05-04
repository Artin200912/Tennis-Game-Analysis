import cv2

# so this func is basically creating a list of frames from a video file
def read_video(video_path):
    # it creates a cv2 capture instance of the video_path
    cap = cv2.VideoCapture(video_path)
    frames = [] #it creates an empty list for the frames
    while True: 
        # when there is a return value, the frame is created as the cap.read() because it reads in the video frames
        ret, frame = cap.read()
        if not ret: # when there is no return value anymore, we break out the loop
            break
        frames.append(frame) # and we add the frame to the frames list.
    cap.release()
    return frames

# this function is used to save that video file and it takes an output_frames list and a path for saving the output
def save_video(output_video_frames, output_video_path):
    # we write the video with this format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # we determine ther output_path, fourcc, frames_per_sec and the shape of it
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames: # and it do this for each frame.  and writes out each frame to the output
        out.write(frame)
    out.release()