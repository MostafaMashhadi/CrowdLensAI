import os
import numpy as np
from ultralytics import YOLO
from utils import get_path_from_root

# Optional mapping of common YOLO class names to numerical IDs.
# Helps in automatically setting "person" class detection only.
YOLO_CLASSES_MAP = {"person": 0}

class Detector:
    """
    Wrapper for YOLO object detection.

    Loads a YOLO model and runs inference on frames.
    By default, configured to detect 'person' class only,
    but can be set to any list/str/int of class IDs.

    Args:
        model_path (str): Path to YOLO model file (.pt)
        conf (float): Confidence threshold
        iou (float): IoU threshold for NMS
        classes (int, str, or list[int]): Class selection
        device (str): Inference device ("cpu", "cuda", "mps")
        img_size (int): Image size for YOLO inference
    """
    def __init__(self, model_path, conf=0.25, iou=0.45, classes=None, device="mps", img_size=1280):
        # Convert path relative to project root → absolute
        model_path = get_path_from_root(model_path)

        # Validate model file existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")
        print(f"[INFO] Loading YOLO model from: {model_path}")

        # Set default class to 'person' if none provided
        if classes is None:
            classes = [YOLO_CLASSES_MAP["person"]]
        elif isinstance(classes, str):
            # Convert comma-separated string → list of int
            classes = [int(c.strip()) for c in classes.split(",") if c.strip().isdigit()]
        elif isinstance(classes, int):
            # Wrap single int into list
            classes = [classes]

        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = classes
        self.device = device
        self.img_size = img_size

        # Load YOLO model
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Run YOLO detection on a single frame.

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Array of detections [x1, y1, x2, y2, conf, cls] (float32).
                        Shape: (N, 6). Empty array if no detections.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            imgsz=self.img_size
        )

        # Handle case: no detections
        if not results or len(results[0].boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)           # Bounding box coords
        confs = boxes.conf.cpu().numpy().reshape(-1, 1).astype(np.float32)  # Confidence scores
        cls_ids = boxes.cls.cpu().numpy().reshape(-1, 1).astype(np.float32) # Class IDs

        return np.hstack((xyxy, confs, cls_ids)).astype(np.float32)

