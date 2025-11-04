import numpy as np

#--------------------------------------------------------- Lateral Offset --------------------------------------------------------#

def compute_lateral_offset(left_fit, right_fit, image_shape, m_per_pix=3.7/700, y_lookahead=None):
    """
    Estimate lateral offset in meters from lane centres.
    The offset is measured at y_lookahead pixels down from the top of the image (e.g. 400 pixels is approx 20m ahead).
    Positive offset means the vehicle is to the right of the lane centre.
    Negative offset means the vehicle is to the left of the lane centre.
    """
    height, width = image_shape[:2]
    if y_lookahead is not None:                                                 # use y_lookahead if given, otherwise use bottom row of image
        y_eval = y_lookahead
    else:
        y_eval = height - 1          

    # Evaluate lane positions at y_eval
    left_x = np.polyval(left_fit, y_eval)                                       # compute x position of left lane at y_eval
    right_x = np.polyval(right_fit, y_eval)                                     # compute x position of right lane at y_eval
    lane_centre = (left_x + right_x) / 2                                        # lane centre is the average x position of both lanes
    camera_centre = width / 2                                                   # assume camera is at the centre of the image width

    # Offset converted from pixels to meters
    lat_offset_pix = camera_centre - lane_centre                                # pixel offset between camera centre and lane centre
    lat_offset_m = lat_offset_pix * m_per_pix                                   # convert pixel offset to meters using scale factor
    lat_offset_m = round(lat_offset_m, 3)
    
    return lat_offset_m

