def get_center_of_bbox(bbox):
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox
    # Calculate the center coordinates
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    # Return the center coordinates as a tuple
    return (center_x, center_y)

def measure_distance(p1, p2):
    # Calculate the Euclidean distance between two points
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox
    # Return the foot position as the center of the bottom edge of the bounding box
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    # Initialize variables to store the closest distance and keypoint index
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    # Iterate through the keypoint indices
    for keypoint_index in keypoint_indices:
        # Get the coordinates of the current keypoint
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        # Calculate the distance between the point and the keypoint
        distance = abs(point[1] - keypoint[1])
        # Update the closest distance and keypoint index if necessary
        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index
    # Return the index of the closest keypoint
    return key_point_ind


def get_height_of_bbox(bbox):
    # Calculate the height of the bounding box
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    # Calculate the absolute differences in x and y coordinates
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

