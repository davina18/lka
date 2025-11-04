import cv2
import numpy as np
from munkres import Munkres
import torch
from sklearn.cluster import DBSCAN
from lanenet.model.lanenet.LaneNet import LaneNet
from utils import compute_lateral_offset

#--------------------------------------------------------- Instantiate Model -------------------------------------------------------#

# Load model
model_path = 'lanenet/log/best_model.pth'
model = LaneNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Instantiate hungarian matcher for lane tracking across frames
munkres = Munkres()

#----------------------------------------------------------- Lane Functions --------------------------------------------------------#

# Fit a 2nd degree polynomial to a set of (x, y) points
def fit_poly(points):
    if len(points) < 5:
        return None
    y, x = zip(*points)
    return np.polyfit(y, x, deg=2)

# Match lanes between previous and current frame using Hungarian algorithm
def match_lanes(prev_lanes, current_lanes):
    cost_matrix = []
    for p in prev_lanes:
        # Compares average x-position of lane p with each lane c in current frame
        row = [abs(np.mean(p[:, 1]) - np.mean(c[:, 1])) for c in current_lanes]
        cost_matrix.append(row)
    # Optimal matched lane pairs stored as (prev_idx, current_idx)
    idxs = munkres.compute(cost_matrix)
    return idxs

# Draws lane curves onto a blank RGB mask and overlays it onto the image
def draw_lane_curve(img, poly, y_min, y_max, color=(0, 255, 0)):
    if poly is None:
        return img

    curve_mask = np.zeros_like(img, dtype=np.uint8)                                 # initialise mask
    y_vals = np.linspace(y_min, y_max, int(y_max - y_min))                          # generate y coords within bounds
    x_vals = np.polyval(poly, y_vals)                                               # evaluate polynomial at y_vals

    # Draw curve on the mask
    for x, y in zip(x_vals, y_vals):
        x_int = int(np.clip(x, 0, img.shape[1] - 1))                                # clip x to image width
        y_int = int(np.clip(y, 0, img.shape[0] - 1))                                # clip y to image height
        cv2.circle(curve_mask, (x_int, y_int), 6, color, -1)               

    # Overlay mask onto original image
    img = cv2.addWeighted(img, 1.0, curve_mask, 0.6, 0)                             # overlay with transparency
    return img

#----------------------------------------------------------- Scoring Functions -----------------------------------------------------#

# Compute confidence score for a lane mask using probability map and temporal agreement
def compute_confidence(prob_map, lane_mask, current_fit=None, prev_fit=None, a=1.0, b=3.0):
    """
    a: weight for probability map score
    b: penalty for temporal drift
    """
    # Multiply probability map by binary lane mask to get weighted confidence
    weighted_sum = np.sum(prob_map * lane_mask)
    total_pixels = np.sum(lane_mask)

    # Avoid division by zero
    if total_pixels == 0:
        return 0.0
    
    prob_map_score = weighted_sum / total_pixels

    # Temporal consistency penalty
    if current_fit is not None and prev_fit is not None:
        temporal_drift = np.linalg.norm(np.array(current_fit) - np.array(prev_fit))
    else:
        temporal_drift = 0.0

    # Final confidence
    raw_score = a * prob_map_score - b * temporal_drift
    confidence = max(0.0, min(1.0, raw_score))
    return round(confidence, 3)


# Scores a lane cluster based on horizontal proximity to image centre and vertical length
def score_cluster(cluster, image_centre, image_height):
    x_centre = np.mean(cluster[:, 1])                                               # mean x position of the cluster
    y_span = np.ptp(cluster[:, 0])                                                  # compute vertical length of the cluster
    x_score = abs(x_centre - image_centre) / image_centre                           # normalise horizontal distance from centre
    y_score = y_span / image_height                                                 # normalize vertical length
    score = x_score - 0.8 * y_score                                                 # lower score is better
    return score

#---------------------------------------------------------- Ego Lane Detection -----------------------------------------------------#

# Create binary mask of all lane pixels detected by the model
def all_lane_pixels(img):
    # Resize frame to model input size and normalise to [-1, 1]
    input_img = cv2.resize(img, (512, 256)) / 127.5 - 1.0
    input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).unsqueeze(0).float()

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        # Probability map where each pixels value indicates how likely it is to be part of any lane in the frame
        binary_seg = output['binary_seg_pred']
        prob_map = binary_seg.squeeze().cpu().numpy()
        # Pixels with confidence > 0.5 are lane pixels (True) and the rest are background pixels (False)
        binary_mask = prob_map > 0.5

    # Resize binary mask and probability mask back to original image size
    binary_mask = cv2.resize(binary_mask.astype(np.uint8), (img.shape[1], img.shape[0]))
    prob_map = cv2.resize(prob_map.astype(np.float32), (img.shape[1], img.shape[0]))

    return binary_mask, prob_map

# Extracts left and right ego lane curves from a binary mask and overlays them on the image
def extract_ego_lanes(binary_mask, image, eps=10, min_samples=50):

    # Get all lane pixels from the binary mask
    lane_points = np.column_stack(np.where(binary_mask > 0))

    # Cluster lane pixels using DBSCAN based on spatial proximity
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(lane_points)
    labels = clustering.labels_
    clusters = [lane_points[labels == i] for i in set(labels) if i != -1]

    # Score and sort clusters based on ego-lane likelihood (lower score is better)
    image_centre = binary_mask.shape[1] // 2    
    image_height = binary_mask.shape[0]         
    scored_clusters = sorted(
    enumerate(clusters),
    key=lambda x: score_cluster(x[1], image_centre, image_height)
    )

    # Select the top two scoring clusters as the ego lanes
    ego_idxs = [scored_clusters[0][0], scored_clusters[1][0]]                       # extract their cluster indices
    ego_clusters = [clusters[i] for i in ego_idxs]                                  # extract their cluster points
    ego_polys = [fit_poly(clusters[i]) for i in ego_idxs]                           # fit a 2nd order polynomial to ego cluster

    # Determine left/right ordering based on horizontal position
    ego_centres = [np.mean(clusters[i][:, 1]) for i in ego_idxs]                    # mean x position of the cluster
    if ego_centres[0] > ego_centres[1]:
        ego_polys = ego_polys[::-1]                                                 # swap to ensure left lane is first

    # Compute maximum vertical bounds across both ego clusters
    y_min = min(np.min(c[:, 0]) for c in ego_clusters)                              # topmost y coord across both clusters
    y_max = max(np.max(c[:, 0]) for c in ego_clusters)                              # bottommost y coord across both clusters

    # Draw ego lanes with vertical bounds
    image = draw_lane_curve(image, ego_polys[0], y_min, y_max, color=(0, 255, 0))   # left = green
    image = draw_lane_curve(image, ego_polys[1], y_min, y_max, color=(255, 0, 0))   # right = blue

    return image, ego_polys, ego_clusters

#-------------------------------------------------------- Overlay Lanes on Frame ---------------------------------------------------#

def overlay_lanes_on_frame(frame):
    binary_mask, prob_map = all_lane_pixels(frame)                                  # get binary mask of all lane pixels detected by LaneNet
    ego_overlay, ego_polys, ego_clusters = extract_ego_lanes(binary_mask, frame)    # get ego overlay image

    # Create binary masks for the ego clusters
    left_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for y, x in ego_clusters[0]:
        left_mask[int(y), int(x)] = 1                                               # set pixels in the left lane cluster to 1
    right_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for y, x in ego_clusters[1]:
        right_mask[int(y), int(x)] = 1                                              # set pixels in the right lane cluster to 1

    # Compute confidence scores
    left_conf = compute_confidence(prob_map, left_mask)
    right_conf = compute_confidence(prob_map, right_mask)

    # Compute lateral offset
    lat_offset_m = compute_lateral_offset(ego_polys[0], ego_polys[1], frame.shape, y_lookahead=400)

    # HUD text
    status_text = f"Left: {'YES' if left_conf > 0.6 else 'NO'} ({left_conf:.2f}) | Right: {'YES' if right_conf > 0.6 else 'NO'} ({right_conf:.2f})"
    cv2.putText(ego_overlay, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return ego_overlay, left_conf, right_conf, lat_offset_m
