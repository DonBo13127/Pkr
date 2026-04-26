"""
vision/yolo_detector.py - YOLOv8 Detection Module

High-performance object detection for poker table elements:
- Cards (playing cards on table and in player hands)
- Chips (stacks and individual chips)
- Players (seated players at the table)
- Dealer button (position marker)
- Pot area (center pot, side pots)
- UI elements (bet amounts, action buttons)

GPU-optimized with batch processing support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Will use mock/dummy mode if not available

logger = logging.getLogger(__name__)


class DetectionClass(Enum):
    """Detection classes for poker table objects."""
    CARD = 0
    CHIP = 1
    PLAYER = 2
    DEALER_BUTTON = 3
    POT = 4
    UI_BET = 5
    UI_ACTION = 6
    TABLE = 7


@dataclass
class DetectionResult:
    """
    Result of a single detection.
    
    Attributes:
        class_id: Integer class identifier
        class_name: Human-readable class name
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence [0, 1]
        mask: Optional segmentation mask (from SAM2)
        track_id: Optional tracking ID (assigned by tracker)
        timestamp: Frame timestamp when detected
    """
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    mask: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    timestamp: float = 0.0
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> int:
        """Return area of bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))
    
    @property
    def aspect_ratio(self) -> float:
        """Return aspect ratio (width/height)."""
        x1, y1, x2, y2 = self.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return w / h
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "center": self.center,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
        }


class YOLODetector:
    """
    YOLOv8-based detector for poker table elements.
    
    Features:
    - GPU acceleration (CUDA) when available
    - Configurable confidence thresholds per class
    - Batch processing for efficiency
    - Fallback to CPU if GPU unavailable
    - Mock mode for testing without model
    """
    
    # Default class names mapping
    CLASS_NAMES = {
        0: "card",
        1: "chip",
        2: "player",
        3: "dealer_button",
        4: "pot",
        5: "ui_bet",
        6: "ui_action",
        7: "table",
    }
    
    # Per-class confidence thresholds
    DEFAULT_THRESHOLDS = {
        "card": 0.45,
        "chip": 0.40,
        "player": 0.50,
        "dealer_button": 0.35,
        "pot": 0.40,
        "ui_bet": 0.35,
        "ui_action": 0.35,
        "table": 0.60,
    }
    
    # NMS IoU threshold
    NMS_THRESHOLD = 0.45
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        conf_threshold: Optional[Dict[str, float]] = None,
        nms_threshold: float = NMS_THRESHOLD,
        use_mock: bool = False,
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
            device: Device to use ("cuda", "cpu", "auto")
            conf_threshold: Per-class confidence thresholds
            nms_threshold: Non-maximum suppression IoU threshold
            use_mock: If True, use mock detector for testing
        """
        self.model_path = model_path
        self.nms_threshold = nms_threshold
        self.use_mock = use_mock
        
        # Set confidence thresholds
        self.conf_thresholds = {**self.DEFAULT_THRESHOLDS}
        if conf_threshold:
            self.conf_thresholds.update(conf_threshold)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if self._check_cuda() else "cpu"
        else:
            self.device = device
        
        # Load model
        self.model = None
        if not use_mock and YOLO is not None:
            self._load_model(model_path)
        elif use_mock:
            logger.info("Using mock YOLO detector (no real inference)")
        else:
            logger.warning("YOLO not available, detections will be empty")
        
        logger.info(f"YOLODetector initialized on device: {self.device}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        if YOLO is None:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self, model_path: Optional[str]) -> None:
        """Load YOLOv8 model."""
        if YOLO is None:
            logger.error("ultralytics package not installed")
            return
        
        try:
            if model_path:
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom model: {model_path}")
            else:
                # Use default YOLOv8n for demo (should be replaced with trained model)
                self.model = YOLO("yolov8n.pt")
                logger.info("Loaded default YOLOv8n model (replace with poker-trained model)")
            
            # Move to device
            if self.device == "cuda":
                self.model.to("cuda")
                logger.info("Model moved to CUDA")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Run detection on an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            classes: Filter to specific class IDs (None = all)
            min_confidence: Override global confidence threshold
            
        Returns:
            List of DetectionResult objects
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detector")
            return []
        
        if self.use_mock:
            return self._mock_detect(image, classes, min_confidence)
        
        if self.model is None:
            logger.warning("No model loaded, returning empty detections")
            return []
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=min_confidence or min(self.conf_thresholds.values()),
                iou=self.nms_threshold,
                classes=classes,
                verbose=False,
            )
            
            # Parse results
            detections = []
            result = results[0]  # Single image
            
            boxes = result.boxes
            if boxes is None:
                return []
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract data
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                
                # Check per-class threshold
                threshold = self.conf_thresholds.get(cls_name, 0.3)
                if conf < threshold:
                    continue
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Create detection result
                detection = DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    timestamp=0.0,  # Will be set by pipeline
                )
                detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _mock_detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Mock detection for testing without real model.
        Generates synthetic detections based on image analysis.
        """
        detections = []
        h, w = image.shape[:2]
        
        # Simulate card detection (white rectangles)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            
            # Filter for card-like objects
            if 2000 < area < 50000:
                aspect = bw / max(1, bh)
                if 0.5 < aspect < 0.8:
                    detections.append(DetectionResult(
                        class_id=0,
                        class_name="card",
                        bbox=(x, y, x + bw, y + bh),
                        confidence=0.75,
                    ))
        
        # Simulate chip detection (circular objects)
        edges = cv2.Canny(gray, 50, 150)
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        if circles is not None:
            for circle in circles[0]:
                x, y, r = map(int, circle)
                detections.append(DetectionResult(
                    class_id=1,
                    class_name="chip",
                    bbox=(x - r, y - r, x + r, y + r),
                    confidence=0.65,
                ))
        
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        classes: Optional[List[int]] = None,
    ) -> List[List[DetectionResult]]:
        """
        Run detection on a batch of images.
        
        Args:
            images: List of input images
            classes: Filter to specific class IDs
            
        Returns:
            List of detection lists (one per image)
        """
        if not images:
            return []
        
        if self.use_mock or self.model is None:
            return [self.detect(img, classes) for img in images]
        
        try:
            # Batch inference
            results = self.model(images, conf=0.3, iou=self.nms_threshold, classes=classes, verbose=False)
            
            all_detections = []
            for idx, result in enumerate(results):
                detections = []
                boxes = result.boxes
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                        threshold = self.conf_thresholds.get(cls_name, 0.3)
                        
                        if conf >= threshold:
                            x1, y1, x2, y2 = map(int, xyxy)
                            detections.append(DetectionResult(
                                class_id=cls_id,
                                class_name=cls_name,
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                            ))
                
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [self.detect(img, classes) for img in images]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "has_model": self.model is not None,
            "use_mock": self.use_mock,
            "class_names": self.CLASS_NAMES,
            "confidence_thresholds": self.conf_thresholds,
            "nms_threshold": self.nms_threshold,
        }
