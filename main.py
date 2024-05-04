from utils.video_utils import read_video, save_video
from utils.bbox_utils import *
from utils.draw_player_ststs_utils import *
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court.mini_court import *
import pandas as pd
from copy import deepcopy

def main(video_path):
    video_path = video_path
    video_frames = read_video(video_path)

    player_tracker = PlayerTracker('yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')
    mini_court = MiniCourt(video_frames[0])

    court_line_detector = CourtLineDetector('models/keypoints_model.pth')
    court_line_detections = court_line_detector.predict(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(court_line_detections, player_detections)

    ball_tracker = BallTracker('models/yolo_5.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/ball_tracker.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                              ball_detections,
                                                                                                          court_line_detections)

    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24 # Assuming 24fps

        # Assuming functions measure_distance, convert_pixel_distance_to_meters, and constants.DOUBLE_LINE_WIDTH are defined elsewhere
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court())

        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                              ball_mini_court_detections[start_frame][1]))

        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']

    # Assuming functions draw_player_stats and draw_points_on_mini_court are defined elsewhere
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_line_detections)

    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame number : {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 2)

    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    video_path = input('Enter a path:')
    main(video_path)
