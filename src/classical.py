import cv2
import numpy as np
import os
import csv
from utils import compute_lateral_offset

#--------------------------------------------------------- ROI -------------------------------------------------------------#

def crop_roi(frame, crop_ratio=0.5):
    """
    Crop out the upper portion of the frame based on crop_ratio to focus on the road area.
    """
    height = frame.shape[0]
    cutoff_height = int(height * crop_ratio)
    roi = frame[cutoff_height:, :]
    return roi

#------------------------------------------------------ Thresholding -------------------------------------------------------#

def hls_sobel_thresholding(roi, s_thresh=(120, 255), sobel_thresh=(70, 255)):
    """
    Convert to HLS. 
    Threshold S channel for lane paint. 
    Threshold Sobel x gradient to emphasise vertical edges. 
    Combine thresholded masks.
    """
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Binary mask of S channel that falls within s_thresh
    s_binary = cv2.inRange(s_channel, s_thresh[0], s_thresh[1])

    # Compute horizontal (x-direction) gradient of the L channel using 3x3 Sobel kernel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)                                                # take abs value
    sobelx = np.uint8(255 * sobelx / np.max(sobelx))                            # normalise to [0, 255] range
    grad_binary = cv2.inRange(sobelx, sobel_thresh[0], sobel_thresh[1])         # binary mask of gradient

    # Combine colour (s_binary) and gradient (grad_binary) masks
    combined_binary = cv2.bitwise_or(s_binary, grad_binary)
    return combined_binary

#--------------------------------------------------------- IPM -------------------------------------------------------------#

def ipm(image,
        trapezoid_top_y=0.0, trapezoid_bottom_y=0.9,
        trapezoid_top_width=(0.2, 0.8), trapezoid_bottom_width=(0.0, 1.0),
        rectangle_top_width=(0.0, 1.0)):
    """
    Use inverse perspective mapping (IPM) to get a birds eye view of the road.
    src_points: 4 points in the original image outlining the lane area (they form a trapezoid).
    dst_points: 4 points that we want the src_points to map to in the top-down view (they form a rectangle)
    """
    height, width = image.shape[:2]
    
    # Source trapezoid
    src_points = np.float32([
        [width * trapezoid_top_width[0], height * trapezoid_top_y],             # top left corner
        [width * trapezoid_top_width[1], height * trapezoid_top_y],             # top right corner
        [width * trapezoid_bottom_width[1], height * trapezoid_bottom_y],       # bottom right corner
        [width * trapezoid_bottom_width[0], height * trapezoid_bottom_y]        # bottom left corner
    ])

    # Destination rectangle (flipped vertically)
    dst_points = np.float32([
        [width * rectangle_top_width[0], 0],                                    # top left becomes bottom left
        [width * rectangle_top_width[1], 0],                                    # top right becomes bottom right
        [width * rectangle_top_width[1], height],                               # bottom right becomes top right
        [width * rectangle_top_width[0], height]                                # bottom left becomes top left
    ])

    # Perspective transformation matrix from src_points to dst_points
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (width, height))

    # Inverse transformation for video overlay
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    return warped, M, Minv

#------------------------------------------------- Extract Lane Pixels -----------------------------------------------------#

def extract_lane_pixels(binary_warped, margin=80, min_pix=50, window_height=10):
    """
    Perform upwards sliding window search to extract lane pixels from a binary warped mask.
    Returns x and y coordinates of pixels detected along the left and right lanes.
    """
    # Sum pixel values vertically (along y-axis) in bottom half of the image to get a 1D x-axis histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    # Convert binary_warped to RGB representation
    debug_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Split histogram into left and right halves
    midpoint = histogram.shape[0] // 2
    # Find x-position of peak in the left half (bottom of the left lane)
    leftx_base = np.argmax(histogram[:midpoint])
    # Find x-position of peak in the right half (bottom of the right lane)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Compute number of windows
    image_height = binary_warped.shape[0]
    no_windows = image_height // window_height

    # Identify nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = nonzero

    # Current x position in left and right lanes
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Indices of detected pixels in left and right lanes
    left_lane_idxs = []
    right_lane_idxs = []

    for window in range(no_windows):
        # Vertical boundaries of current sliding window
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # Horizontal boundaries of left sliding window
        win_xleft_left = leftx_current - margin
        win_xlef_right = leftx_current + margin

        # Horizontal boundaries of right sliding window
        win_xright_left = rightx_current - margin
        win_xright_right = rightx_current + margin

        # Draw windows for visualisation
        cv2.rectangle(debug_img, (win_xleft_left, win_y_low), (win_xlef_right, win_y_high), (0,255,0), 2)
        cv2.rectangle(debug_img, (win_xright_left, win_y_low), (win_xright_right, win_y_high), (0,255,0), 2)

        # Identify nonzero pixel indices within left and right windows
        current_left_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_left) & (nonzerox < win_xlef_right)).nonzero()[0]
        current_right_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_left) & (nonzerox < win_xright_right)).nonzero()[0]

        # Append indices
        left_lane_idxs.append(current_left_idxs)
        right_lane_idxs.append(current_right_idxs)

        # If enough lane pixels are found, recentre the next window around their mean x-position
        if len(current_left_idxs) > min_pix:
            leftx_current = int(np.mean(nonzerox[current_left_idxs]))
        if len(current_right_idxs) > min_pix:
            rightx_current = int(np.mean(nonzerox[current_right_idxs]))

    # Concatenate indices from each window into an array
    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    # Extract x and y coordinates of all pixels classified as part of the lanes
    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]
    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    # Create binary lane mask
    lane_mask = np.zeros_like(binary_warped)
    # Mark lane pixels as 1
    lane_mask[lefty, leftx] = 1
    lane_mask[righty, rightx] = 1

    return leftx, lefty, rightx, righty, debug_img, lane_mask

#----------------------------------------------------- Fit Lane Pixels -----------------------------------------------------#

def fit_lane_curve(x, y, order=2, min_pixels=100, min_radius=150):
    """
    Fit a polynomial to lane pixels.
    Rejects fits with too few pixels or small curve radius (sharp turn).
    Returns polynomial coefficients or None.
    """
    # Reject fit if there's too few pixels
    if len(x) < min_pixels:
        print("Fit rejected due to too few pixels")
        return None

    # Fit polynomial
    fit = np.polyfit(y, x, order)

    # Reject fit if its a tight curve (sharp turn)
    y_eval = np.max(y)                                                          # vertical position (in pixels) at which we evaluate the curvature
    A, B, C = fit                                                               # polynomial coefficients (A*y^2 + B*y + C)
    radius = ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A)                     # curvature radius
    if radius < min_radius:
        print(f"Fit rejected due to tight curve. Radius is {radius}")
        return None

    return fit

#------------------------------------------------------- Curve Overlay Mask ------------------------------------------------------#

# Draws lane curves onto a blank RGB mask
def draw_curve_mask(left_fit, right_fit, shape, left_conf, right_conf, conf_thresh=0.6):
    curve_mask = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)              # initialise mask
    yvals = np.linspace(0, shape[0] - 1, shape[0])                              # image height y values

    # Choose colors based on confidence
    left_color = (0, 255, 0) if left_conf > conf_thresh else (128, 128, 128)    # green or gray
    right_color = (255, 0, 0) if right_conf > conf_thresh else (128, 128, 128)  # blue or gray

    # Draw left curve
    for x, y in zip(np.polyval(left_fit, yvals), yvals):
        x_int = int(np.clip(x, 0, shape[1] - 1))                                # clip x to image width
        y_int = int(np.clip(y, 0, shape[0] - 1))                                # clip y to image height
        cv2.circle(curve_mask, (x_int, y_int), 10, left_color, -1)
    # Draw right curve
    for x, y in zip(np.polyval(right_fit, yvals), yvals):
        x_int = int(np.clip(x, 0, shape[1] - 1))                                # clip x to image width
        y_int = int(np.clip(y, 0, shape[0] - 1))                                # clip y to image height
        cv2.circle(curve_mask, (x_int, y_int), 10, right_color, -1)

    return curve_mask

#------------------------------------------------------- Confidence --------------------------------------------------------#

def compute_confidence(x, y, fit, prev_fit=None):
    """
    Compute confidence score in range [0,1], based on pixel count, fit residual, and temporal consistency.
    """
    # If no polynomial was fitted
    if fit is None or len(x) == 0:
        return 0.0

    # Pixel count score in range [0, 1] (500 < returns a score of 1)
    pixel_score = min(len(x) / 500, 1.0)

    # Fit residual score
    y_pred = np.polyval(fit, y)                                                 # predicted x from fit at each y
    residual = np.mean((x - y_pred) ** 2)                                       # MSE between predicted and gt x
    residual_score = 1.0 / (1.0 + residual / 1000)                              # lower residual means a higher score

    # Temporal consistency score
    if prev_fit is not None:
        diff = np.linalg.norm(np.array(fit) - np.array(prev_fit))               # L2 distance between current frame and previous frame fit
        temporal_score = 1.0 / (1.0 + diff)                                     # smaller difference means a higher score
    else:
        temporal_score = 1.0

    # Weighted average
    confidence = 0.4 * pixel_score + 0.3 * residual_score + 0.3 * temporal_score
    confidence = round(confidence, 3)
    return confidence

#--------------------------------------------------- Temporal Smoothing ---------------------------------------------------#

def smooth_fit(current_fit, prev_fit, alpha=0.7):
    """
    Exponential smoothing of polynomial coefficients.
    Fallback to previous fit if current fit is None.
    """
    if current_fit is None:
        return prev_fit
    if prev_fit is None:
        return current_fit
    smoothed_fit = alpha * np.array(current_fit) + (1 - alpha) * np.array(prev_fit)
    return smoothed_fit

#------------------------------------------------------ Overlay Lanes on Frame ---------------------------------------------------#

def overlay_lanes_on_frame(frame):
    # Track previous fits
    global prev_left_fit, prev_right_fit

    # Apply transformations to frame
    roi = crop_roi(frame, crop_ratio=0.5)
    binary_mask = hls_sobel_thresholding(roi, s_thresh=(120, 255), sobel_thresh=(70, 255))
    warped_binary, _, _ = ipm(binary_mask)

    # Extract lane pixels and fit lane curves
    leftx, lefty, rightx, righty, debug_img, _ = extract_lane_pixels(warped_binary, window_height=12)
    left_fit = fit_lane_curve(leftx, lefty)
    right_fit = fit_lane_curve(rightx, righty)

    # Compute confidence scores
    left_conf = compute_confidence(leftx, lefty, left_fit, prev_left_fit)
    right_conf = compute_confidence(rightx, righty, right_fit, prev_right_fit)

    # Apply temporal smoothing if confidence is high enough
    if left_conf < 0.6:
        left_fit = prev_left_fit
    else:
        left_fit = smooth_fit(left_fit, prev_left_fit)
    if right_conf < 0.6:
        right_fit = prev_right_fit
    else:
        right_fit = smooth_fit(right_fit, prev_right_fit)

    # Update previous fits
    prev_left_fit = left_fit
    prev_right_fit = right_fit

    if left_fit is not None and right_fit is not None:
        # Generate lane curve mask in warped view
        curve_mask = draw_curve_mask(left_fit, right_fit, warped_binary.shape[:2], left_conf, right_conf)
        # Get inverse perspective transformation
        _, _, Minv = ipm(warped_binary)
        # Unwarp curve mask
        unwarped_curve_mask = cv2.warpPerspective(curve_mask, Minv, (roi.shape[1], roi.shape[0]))
        # Overlay curves on the roi section of the original frame
        overlay = frame.copy()
        overlay[int(overlay.shape[0] * 0.5):, :] = cv2.addWeighted(roi, 0.6, unwarped_curve_mask, 1.0, 0)   
        # Compute lateral offset
        lat_offset_m = compute_lateral_offset(left_fit, right_fit, frame.shape, y_lookahead=400)

        # HUD text
        status_text = f"Left: {'YES' if left_conf > 0.6 else 'NO'} ({left_conf:.2f}) | Right: {'YES' if right_conf > 0.6 else 'NO'} ({right_conf:.2f})"
        cv2.putText(overlay, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
    return overlay, left_conf, right_conf, lat_offset_m


#-------------------------------------------------- Video Overlay -------------------------------------------------------#

# Track previous fits
prev_left_fit = None
prev_right_fit = None

if __name__ == "__main__":
    # Define input and output directories
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_dir = os.path.join(project_root, 'LKA/data/CULane', '1')
    output_dir = os.path.join(project_root, 'LKA/classical_outputs', '7')
    os.makedirs(output_dir, exist_ok=True) 

    # Get sorted list of frames in the input directory
    frames = sorted(f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png')))

    # Setup csv file
    csv_path = os.path.join(output_dir, "lane_metrics.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_id', 'left_detected', 'right_detected', 'left_conf', 'right_conf', 'lat_offset_m'])

    # Process each frame
    for frame_id, filename in enumerate(frames):
        frame_path = os.path.join(input_dir, filename)
        frame = cv2.imread(frame_path)
        overlay, left_conf, right_conf, lat_offset_m = overlay_lanes_on_frame(frame)
        left_detected = int(left_conf > 0.6)
        right_detected = int(right_conf > 0.6)
        csv_writer.writerow([frame_id, left_detected, right_detected, round(left_conf, 2), round(right_conf, 2), lat_offset_m])
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, overlay)
        cv2.imshow("Lane Overlay", overlay)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    csv_file.close()
