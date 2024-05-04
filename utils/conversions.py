def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    # Calculate the conversion factor from pixels to meters
    conversion_factor = refrence_height_in_meters / refrence_height_in_pixels
    # Convert the pixel distance to meters using the conversion factor
    meters_distance = pixel_distance * conversion_factor
    # Return the converted distance in meters
    return meters_distance

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    # Calculate the conversion factor from meters to pixels
    conversion_factor = refrence_height_in_pixels / refrence_height_in_meters
    # Convert the meters distance to pixels using the conversion factor
    pixel_distance = meters * conversion_factor
    # Return the converted distance in pixels
    return pixel_distance
