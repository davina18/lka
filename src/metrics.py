import csv
import time
import numpy as np
from classical import overlay_lanes_on_frame
import os
import cv2

#--------------------------------------------------------- CSV Loading ---------------------------------------------------------#

# Load metrics from CSV file
def load_csv_labels(path):
    labels = {}                             
    with open(path, 'r') as f:
        reader = csv.DictReader(f)          
        for row in reader:
            clip_id = int(row['clip_id'])
            frame_id = int(row['frame_id'])
            labels[(clip_id, frame_id)] = {
                'left_detected': int(row['left_detected']),
                'right_detected': int(row['right_detected']),
                'lat_offset_m': float(row['lat_offset_m'])
            }
    return labels

#------------------------------------------------------ Metric Computation -----------------------------------------------------#                           

# Latency test
def measure_latency(frame_dir):
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png'))
    processing_times = []                            
    for filename in frames:
        img = cv2.imread(os.path.join(frame_dir, filename))  
        start = time.time()                                                                         # start timer
        overlay_lanes_on_frame(img)                                                                 # run overlay function
        processing_times.append(time.time() - start)                                                # record time passed
    print(f"Average latency: {1000 * np.mean(processing_times):.2f} ms/frame")

#------------------------------------------------------ Evaluation Function ----------------------------------------------------#

# Evaluation function
def evaluate(preds):
    total = len(preds)                                                                              # number of frames
    correct_left = correct_right = 0                                                                # counters for detected sides
    lateral_offsets = []                    

    for pred in preds.values():                 

        # Side detection counts (not accuracy without GT)
        if pred['left_detected']:
            correct_left += 1                                                                       # increment if left lane detected
        if pred['right_detected']:
            correct_right += 1                                                                      # increment if right lane detected

        lateral_offsets.append(pred['lat_offset_m'])

    # Mean absolute lateral offset in meters
    mean_offset_m = np.mean(np.abs(lateral_offsets))

    # Print metrics
    print("#-----Prediction-Only Metrics-----#")
    print(f"Left detection rate:        {100 * correct_left / total:.2f}%")                         # % frames with left lane detected
    print(f"Right detection rate:       {100 * correct_right / total:.2f}%")                        # % frames with right lane detected
    print(f"Side detection accuracy:    {100 * (correct_left + correct_right) / (2 * total):.2f}%") # combined left/right accuracy
    print(f"Mean lateral offset:        {mean_offset_m:.3f} m")                                     # mean lateral offset in meters
    print(f"Std-dev of lateral offset:  {np.std(lateral_offsets):.3f} m")                           # temporal stability
    print(f"Frames evaluated:           {total}")                                                   # total frames processed

#---------------------------------------------------------- Run Evaluation -----------------------------------------------------#

# Define paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
pred_csv_path = os.path.join(project_root, 'LKA/video_outputs/TuSimple', 'dl_metrics.csv')
frame_dir = os.path.join(project_root, 'LKA/data/TuSimple', '1')     

# Run evaluation
if __name__ == "__main__":
    preds = load_csv_labels(pred_csv_path)     
    evaluate(preds)          
    measure_latency(frame_dir=frame_dir)  
