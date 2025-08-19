import os
from boxmot import create_tracker
from utils import get_path_from_root

class Tracker:
    """
    Tracker wrapper for ByteTrack (via BoxMOT).

    Optionally integrates a Re-ID model for improved ID persistence.

    Args:
        reid_model (str): Path to Re-ID model weights (optional)
        device (str): "cpu", "cuda", or "mps"
    """
    def __init__(self, reid_model=None, device="mps"):
        tracker_type = "bytetrack"
        tracker_cfg_path = get_path_from_root("src/bytetrack.yaml")

        # Load ReID weights if provided
        reid_model_path = None
        if reid_model:
            reid_model_path = get_path_from_root(reid_model)
            if os.path.exists(reid_model_path):
                print(f"[INFO] Loaded ReID model from: {reid_model_path}")
            else:
                print(f"[WARNING] ReID model not found: {reid_model_path}, continuing without ReID...")
                reid_model_path = None

        # Initialize tracker
        self.tracker = create_tracker(
            tracker_type,
            tracker_config=tracker_cfg_path,
            reid_weights=reid_model_path,
            device=device,
            half=True
        )

    def update(self, detections, frame):
        """
        Update tracker with latest detections.

        Args:
            detections (np.ndarray): [x1, y1, x2, y2, conf, cls] (N, 6)
            frame (np.ndarray): Current BGR frame.
        Returns:
            list: Tracker outputs (varies based on ByteTrack settings).
        """
        return self.tracker.update(detections, frame)


