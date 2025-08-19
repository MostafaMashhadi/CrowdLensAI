import cv2
import time
import json
import numpy as np
from datetime import datetime
from utils import load_config, get_timestamped_filename, Logger
from detector import Detector
from tracker import Tracker

# ------------------------
# Helper functions
# ------------------------
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    Used for appearance matching in Re-ID.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def draw_info_block(frame, lines, x, y, font=cv2.FONT_HERSHEY_SIMPLEX,
                    scale=0.8, color=(255,255,255), thickness=2,
                    bg_color=(0,0,0), alpha=0.5, line_spacing=5):
    """
    Draw a semi-transparent information block on the video frame.

    Args:
        frame: Video frame (NumPy array)
        lines: List of strings to display
        x, y: Top-left corner coordinates
        font, scale, color, thickness: OpenCV text settings
        bg_color: Background color of block (BGR)
        alpha: Transparency of background
        line_spacing: Vertical space between lines
    """
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    max_width = max(w for (w, h) in text_sizes)
    total_height = sum(h for (w, h) in text_sizes) + line_spacing*(len(lines)-1)

    sub_img = frame[y:y+total_height+10, x:x+max_width+10]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        overlay = sub_img.copy()
        overlay[:] = bg_color
        cv2.addWeighted(overlay, alpha, sub_img, 1-alpha, 0, sub_img)

    y_offset = y + text_sizes[0][1] + 2
    for line, (w, h) in zip(lines, text_sizes):
        cv2.putText(frame, line, (x+5, y_offset), font, scale, color, thickness)
        y_offset += h + line_spacing

# ------------------------
# Main processing loop
# ------------------------
def main():
    cfg = load_config()

    # Initialize YOLO detector
    detector = Detector(
        model_path=cfg["yolo_model"],
        device=cfg.get("device", "mps"),
        iou=cfg.get("iou_threshold", 0.5),
        conf=cfg.get("conf_threshold", 0.5),
        classes=cfg.get("classes", None),
        img_size=1920
    )

    # Initialize tracker (ByteTrack + optional Re-ID)
    tracker = Tracker(
        reid_model=cfg.get("reid_model", None),
        device=cfg.get("device", "mps")
    )

    # Open video input (camera or file)
    cap = cv2.VideoCapture(cfg.get("stream_source", 0))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video setup
    out_path = get_timestamped_filename("crowdlens", "mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    Logger.info(f"Saving output video to: {out_path}")

    # Tracking and counting state variables
    unique_ids, temp_seen, tid_to_gid, global_features = set(), {}, {}, {}
    global_id_counter = 0
    GRACE_TIME, SIM_THRESH = 0.5, 0.7  # ID persistence and Re-ID similarity threshold
    MIN_WIDTH, MIN_HEIGHT = 30, 50     # Detection filtering dimensions
    counts_history, last_saved_second = [], -1
    display_id_map, next_display_id = {}, 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            detections = detector.detect(frame)

            # Filter out small detections
            filtered = [det for det in detections if (det[2]-det[0]) >= MIN_WIDTH and (det[3]-det[1]) >= MIN_HEIGHT]
            detections = np.array(filtered, dtype=np.float32) if filtered else np.zeros((0,6), np.float32)

            # Update tracker
            tracked_objs = tracker.update(detections, frame)
            now = time.time()

            # Process each tracked object
            for obj in tracked_objs:
                x1, y1, x2, y2, tid = map(int, obj[:5])
                feature = np.array(obj[5], dtype=np.float32) if len(obj) > 5 and isinstance(obj[5], (np.ndarray, list)) else None

                # Maintain a mapping between tracker IDs and global IDs
                if tid in tid_to_gid:
                    gid = tid_to_gid[tid]
                else:
                    matched_gid = None
                    # Re-ID matching using cosine similarity
                    if feature is not None and feature.size > 0:
                        for g_id, g_feat in global_features.items():
                            if cosine_similarity(feature, g_feat) >= SIM_THRESH:
                                matched_gid = g_id
                                break
                    # If no match found, assign a new global ID
                    if matched_gid is None:
                        global_id_counter += 1
                        matched_gid = global_id_counter
                        if feature is not None and feature.size > 0:
                            global_features[matched_gid] = feature
                    tid_to_gid[tid] = matched_gid
                    gid = matched_gid

                # GRACE_TIME logic to stabilize counting
                if gid not in unique_ids:
                    if gid not in temp_seen:
                        temp_seen[gid] = now
                    elif now - temp_seen[gid] >= GRACE_TIME:
                        unique_ids.add(gid)
                        temp_seen.pop(gid, None)
                else:
                    temp_seen.pop(gid, None)

                # Draw bounding boxes and IDs
                color = (255, 80, 0) if gid in unique_ids else (180, 180, 180)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                if gid in unique_ids:
                    if gid not in display_id_map:
                        display_id_map[gid] = next_display_id
                        next_display_id += 1
                    cv2.putText(frame, f"ID {display_id_map[gid]}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Top-left info block with project name and count
            draw_info_block(
                frame,
                ["CrowdLens AI", f"People count up to now: {len(unique_ids)}"],
                x=10, y=10, scale=1.2
            )

            # Save per-second count history to list
            current_second = int(now)
            if current_second != last_saved_second:
                counts_history.append({
                    "time": datetime.now().isoformat(timespec='seconds'),
                    "count": len(unique_ids)
                })
                last_saved_second = current_second

            # Write frame to output video
            out.write(frame)

    finally:
        # Save resources and write final outputs
        out.release()
        cap.release()
        json_path = get_timestamped_filename("people_count_history", "json")
        with open(json_path, "w") as jf:
            json.dump(counts_history, jf, indent=4)
        Logger.info(f"Video saved: {out_path}")
        Logger.info(f"Count history saved: {json_path}")

if __name__ == "__main__":
    main()
