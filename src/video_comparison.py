import os
import cv2
import csv
from classical import overlay_lanes_on_frame as classical_method
from dl import overlay_lanes_on_frame as dl_method

#--------------------------------------------------------- Initialise ----------------------------------------------------------#

# List of input clip folders
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
clip_folders = [
    os.path.join(project_root, 'LKA/data/TuSimple', '1'),
    os.path.join(project_root, 'LKA/data/TuSimple', '2'),
    os.path.join(project_root, 'LKA/data/TuSimple', '3'),
    os.path.join(project_root, 'LKA/data/TuSimple', '4'),
    os.path.join(project_root, 'LKA/data/TuSimple', '5'),
    os.path.join(project_root, 'LKA/data/TuSimple', '6'),
    os.path.join(project_root, 'LKA/data/TuSimple', '7'),
]

# Output folder
output_dir = os.path.join(project_root, 'LKA/video_outputs')
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, "comparison_video.mp4")

# Initialise video writer
video_writer = None
fps = 2

# Setup CSV file
classical_csv_path = os.path.join(output_dir, "TuSimple", "classical_metrics.csv")
dl_csv_path = os.path.join(output_dir, "TuSimple", "dl_metrics.csv")
classical_csv = open(classical_csv_path, mode='w', newline='')
dl_csv = open(dl_csv_path, mode='w', newline='')
classical_writer = csv.writer(classical_csv)
dl_writer = csv.writer(dl_csv)
header = ['clip_id', 'frame_id', 'left_detected', 'right_detected', 'left_conf', 'right_conf', 'lat_offset_m']
classical_writer.writerow(header)
dl_writer.writerow(header)

#------------------------------------------------ Output Video and CSV File ----------------------------------------------------#

for clip_id, folder in enumerate(clip_folders):
    # Get sorted list of frames in the folder
    frames = sorted(f for f in os.listdir(folder) if f.endswith(('.jpg', '.png')))

    for frame_id, filename in enumerate(frames):
        frame_path = os.path.join(folder, filename)
        frame = cv2.imread(frame_path)

        # Classical method
        classical_overlay, c_left_conf, c_right_conf, c_lat_offset = classical_method(frame.copy())
        c_left_detected = int(c_left_conf > 0.6)
        c_right_detected = int(c_right_conf > 0.6)
        classical_writer.writerow([clip_id, frame_id, c_left_detected, c_right_detected,
                                   round(c_left_conf, 2), round(c_right_conf, 2), c_lat_offset])

        # DL method
        dl_overlay, d_left_conf, d_right_conf, d_lat_offset = dl_method(frame.copy())
        d_left_detected = int(d_left_conf > 0.6)
        d_right_detected = int(d_right_conf > 0.6)
        dl_writer.writerow([clip_id, frame_id, d_left_detected, d_right_detected,
                            round(d_left_conf, 2), round(d_right_conf, 2), d_lat_offset])

        # Resize both outputs to the same size
        height = max(classical_overlay.shape[0], dl_overlay.shape[0])
        width = max(classical_overlay.shape[1], dl_overlay.shape[1])
        classical_overlay = cv2.resize(classical_overlay, (width, height))
        dl_overlay = cv2.resize(dl_overlay, (width, height))

        # Add method labels
        cv2.putText(classical_overlay, "Classical Method", (width - 250, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(dl_overlay, "DL Method", (width - 180, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Stack vertically
        stacked_overlays = cv2.vconcat([classical_overlay, dl_overlay])

        # Initialize video writer on first frame
        if video_writer is None:
            format = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, format, fps, (stacked_overlays.shape[1], stacked_overlays.shape[0]))

        video_writer.write(stacked_overlays)

# Save video
if video_writer:
    video_writer.release()
    print(f"Comparison video saved to: {output_video_path}")
classical_csv.close()
dl_csv.close()
print(f"CSV files saved to: {classical_csv_path} and {dl_csv_path}")