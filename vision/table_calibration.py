"""
vision/table_calibration.py - Dynamic Table Calibration System

Production-grade table layout detection and normalization:
- Automatic table boundary detection
- Player seat position detection relative to table
- Board region and pot region detection
- Normalized coordinate system (0 → 1)
- Adaptive ROI system based on detected table structure
- Layout robustness for different UI themes
- Auto-recalibration when confidence drops or table moves

This module ensures the vision system works across:
- Different screen resolutions
- Various poker client UIs
- Table resizing and repositioning
- Multiple camera angles (for live tables)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
from enum import Enum
from collections import deque
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SeatPosition(Enum):
    """Standard poker seat positions."""
    HERO = 0      # Bottom center (our position)
    UTG = 1       # Under the gun
    MP1 = 2       # Middle position 1
    MP2 = 3       # Middle position 2
    HJ = 4        # Hijack
    CO = 5        # Cutoff
    BTN = 6       # Button
    SB = 7        # Small blind
    BB = 8        # Big blind


@dataclass
class NormalizedBox:
    """
    Bounding box in normalized coordinates (0-1).
    
    Attributes:
        x1, y1, x2, y2: Normalized coordinates [0, 1]
        confidence: Confidence of this region detection
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    
    def __post_init__(self):
        # Clamp to valid range
        self.x1 = max(0.0, min(1.0, self.x1))
        self.y1 = max(0.0, min(1.0, self.y1))
        self.x2 = max(0.0, min(1.0, self.x2))
        self.y2 = max(0.0, min(1.0, self.y2))
        
        # Ensure valid ordering
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates for given image size."""
        x1 = int(self.x1 * width)
        y1 = int(self.y1 * height)
        x2 = int(self.x2 * width)
        y2 = int(self.y2 * height)
        return (x1, y1, x2, y2)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if normalized point is inside this box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def iou(self, other: 'NormalizedBox') -> float:
        """Calculate IoU with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union


@dataclass
class SeatInfo:
    """Information about a player seat."""
    position: SeatPosition
    region: NormalizedBox
    label_region: Optional[NormalizedBox] = None
    chip_region: Optional[NormalizedBox] = None
    card_region: Optional[NormalizedBox] = None
    confidence: float = 1.0
    is_active: bool = True


@dataclass
class TableLayout:
    """
    Complete table layout information.
    
    This defines the spatial structure of the poker table in normalized
    coordinates, allowing the system to work across different resolutions
    and layouts.
    """
    # Main table boundary
    table_bbox: NormalizedBox
    
    # Playing surface regions
    board_region: NormalizedBox  # Community cards area
    pot_region: NormalizedBox    # Pot display area
    
    # Player seats (up to 9 for full ring)
    seats: List[SeatInfo] = field(default_factory=list)
    
    # Additional UI regions
    dealer_button_region: Optional[NormalizedBox] = None
    action_button_region: Optional[NormalizedBox] = None
    bet_input_region: Optional[NormalizedBox] = None
    
    # Metadata
    confidence: float = 1.0
    timestamp: float = 0.0
    layout_version: int = 0  # Incremented on recalibration
    
    # Detection metadata
    table_shape: str = "oval"  # "oval", "rectangle", "custom"
    
    def get_seat_by_position(self, pos: SeatPosition) -> Optional[SeatInfo]:
        """Get seat info by position."""
        for seat in self.seats:
            if seat.position == pos:
                return seat
        return None
    
    def get_seats_in_range(self, start: SeatPosition, count: int) -> List[SeatInfo]:
        """Get consecutive seats starting from position."""
        result = []
        positions = list(SeatPosition)
        start_idx = positions.index(start)
        
        for i in range(count):
            idx = (start_idx + i) % len(positions)
            pos = positions[idx]
            seat = self.get_seat_by_position(pos)
            if seat:
                result.append(seat)
        
        return result
    
    def point_to_seat(self, x: float, y: float) -> Optional[SeatInfo]:
        """Find which seat contains a normalized point."""
        for seat in self.seats:
            if seat.region.contains_point(x, y):
                return seat
        return None
    
    def point_to_region(self, x: float, y: float) -> str:
        """Identify what region a point belongs to."""
        if self.board_region.contains_point(x, y):
            return "board"
        if self.pot_region.contains_point(x, y):
            return "pot"
        
        for seat in self.seats:
            if seat.region.contains_point(x, y):
                return f"seat_{seat.position.name}"
        
        if self.table_bbox.contains_point(x, y):
            return "table"
        
        return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize layout to dictionary."""
        return {
            "table_bbox": {
                "x1": self.table_bbox.x1,
                "y1": self.table_bbox.y1,
                "x2": self.table_bbox.x2,
                "y2": self.table_bbox.y2,
            },
            "seats": [
                {
                    "position": s.position.name,
                    "region": {
                        "x1": s.region.x1, "y1": s.region.y1,
                        "x2": s.region.x2, "y2": s.region.y2,
                    },
                    "confidence": s.confidence,
                    "is_active": s.is_active,
                }
                for s in self.seats
            ],
            "board_region": {
                "x1": self.board_region.x1,
                "y1": self.board_region.y1,
                "x2": self.board_region.x2,
                "y2": self.board_region.y2,
            },
            "pot_region": {
                "x1": self.pot_region.x1,
                "y1": self.pot_region.y1,
                "x2": self.pot_region.x2,
                "y2": self.pot_region.y2,
            },
            "confidence": self.confidence,
            "layout_version": self.layout_version,
            "table_shape": self.table_shape,
        }


class TableCalibrator:
    """
    Dynamic table calibration system.
    
    Features:
    - Automatic table layout detection from frame analysis
    - Normalized coordinate system for resolution independence
    - Adaptive ROI definitions based on detected structure
    - Layout history for stability
    - Auto-recalibration triggers
    - Support for multiple table types/themes
    """
    
    # Default layout templates for common configurations
    LAYOUT_TEMPLATES = {
        "6max_standard": {
            "table_bbox": (0.1, 0.15, 0.9, 0.85),
            "board_region": (0.4, 0.35, 0.6, 0.5),
            "pot_region": (0.45, 0.45, 0.55, 0.55),
            "seats": [
                (SeatPosition.HERO, (0.45, 0.75, 0.55, 0.95)),
                (SeatPosition.SB, (0.25, 0.65, 0.35, 0.8)),
                (SeatPosition.BB, (0.15, 0.5, 0.25, 0.65)),
                (SeatPosition.UTG, (0.15, 0.35, 0.25, 0.5)),
                (SeatPosition.MP1, (0.25, 0.2, 0.35, 0.35)),
                (SeatPosition.CO, (0.65, 0.2, 0.75, 0.35)),
            ],
        },
        "9max_standard": {
            "table_bbox": (0.05, 0.1, 0.95, 0.9),
            "board_region": (0.42, 0.35, 0.58, 0.5),
            "pot_region": (0.46, 0.45, 0.54, 0.55),
            "seats": [
                (SeatPosition.HERO, (0.45, 0.8, 0.55, 0.98)),
                (SeatPosition.SB, (0.3, 0.75, 0.4, 0.9)),
                (SeatPosition.BB, (0.18, 0.65, 0.28, 0.8)),
                (SeatPosition.UTG, (0.1, 0.5, 0.2, 0.65)),
                (SeatPosition.MP1, (0.1, 0.35, 0.2, 0.5)),
                (SeatPosition.MP2, (0.18, 0.2, 0.28, 0.35)),
                (SeatPosition.HJ, (0.3, 0.1, 0.4, 0.25)),
                (SeatPosition.CO, (0.6, 0.1, 0.7, 0.25)),
                (SeatPosition.BTN, (0.72, 0.2, 0.82, 0.35)),
            ],
        },
    }
    
    # Recalibration thresholds
    CONFIDENCE_DROP_THRESHOLD = 0.5
    LAYOUT_CHANGE_IOU_THRESHOLD = 0.7
    RECALIBRATION_INTERVAL = 300  # Frames between periodic checks
    HISTORY_SIZE = 50
    
    def __init__(
        self,
        default_layout: str = "6max_standard",
        enable_auto_detect: bool = True,
        min_confidence: float = 0.6,
    ):
        """
        Initialize table calibrator.
        
        Args:
            default_layout: Default layout template name
            enable_auto_detect: Enable automatic layout detection
            min_confidence: Minimum confidence for accepting detections
        """
        self.default_layout = default_layout
        self.enable_auto_detect = enable_auto_detect
        self.min_confidence = min_confidence
        
        # Current layout
        self.current_layout: Optional[TableLayout] = None
        self.layout_history: deque = deque(maxlen=self.HISTORY_SIZE)
        
        # Calibration state
        self.calibration_count = 0
        self.last_recalibration_frame = 0
        self.frames_since_calibration = 0
        
        # Detection history for stability
        self.detection_history: deque = deque(maxlen=20)
        
        # Confidence tracking
        self.confidence_history: deque = deque(maxlen=30)
        
        logger.info(f"TableCalibrator initialized with default layout: {default_layout}")
    
    def initialize_from_template(self, layout_name: str) -> TableLayout:
        """
        Initialize layout from a predefined template.
        
        Args:
            layout_name: Name of the layout template
            
        Returns:
            Initialized TableLayout
        """
        if layout_name not in self.LAYOUT_TEMPLATES:
            logger.warning(f"Unknown layout '{layout_name}', using default")
            layout_name = self.default_layout
        
        template = self.LAYOUT_TEMPLATES[layout_name]
        
        # Create table bbox
        x1, y1, x2, y2 = template["table_bbox"]
        table_bbox = NormalizedBox(x1, y1, x2, y2, confidence=1.0)
        
        # Create board region
        x1, y1, x2, y2 = template["board_region"]
        board_region = NormalizedBox(x1, y1, x2, y2, confidence=1.0)
        
        # Create pot region
        x1, y1, x2, y2 = template["pot_region"]
        pot_region = NormalizedBox(x1, y1, x2, y2, confidence=1.0)
        
        # Create seats
        seats = []
        for pos, (sx1, sy1, sx2, sy2) in template["seats"]:
            seat_region = NormalizedBox(sx1, sy1, sx2, sy2, confidence=1.0)
            seats.append(SeatInfo(
                position=pos,
                region=seat_region,
                confidence=1.0,
                is_active=True,
            ))
        
        layout = TableLayout(
            table_bbox=table_bbox,
            board_region=board_region,
            pot_region=pot_region,
            seats=seats,
            confidence=1.0,
            timestamp=time.time(),
            layout_version=1,
            table_shape="oval",
        )
        
        self.current_layout = layout
        self.calibration_count += 1
        
        logger.info(f"Initialized layout from template: {layout_name}")
        return layout
    
    def calibrate_from_detections(
        self,
        detections: List[Any],
        image_shape: Tuple[int, int],
        force: bool = False,
    ) -> TableLayout:
        """
        Calibrate table layout from YOLO detections.
        
        Args:
            detections: List of DetectionResult objects
            image_shape: (height, width) of the image
            force: Force recalibration regardless of thresholds
            
        Returns:
            Updated TableLayout
        """
        h, w = image_shape[:2]
        self.frames_since_calibration += 1
        
        # Check if recalibration is needed
        if not force and not self._should_recalibrate():
            if self.current_layout:
                return self.current_layout
        
        logger.info("Performing table calibration...")
        
        # Group detections by class
        table_dets = [d for d in detections if d.class_name == "table"]
        player_dets = [d for d in detections if d.class_name == "player"]
        pot_dets = [d for d in detections if d.class_name == "pot"]
        button_dets = [d for d in detections if d.class_name == "dealer_button"]
        
        # Detect table boundary
        table_bbox = self._detect_table_boundary(table_dets, w, h)
        
        # If no table detected, use fallback
        if table_bbox is None:
            if self.current_layout:
                table_bbox = self.current_layout.table_bbox
            else:
                # Use default template
                return self.initialize_from_template(self.default_layout)
        
        # Detect board region (center of table)
        board_region = self._detect_board_region(table_bbox, detections, w, h)
        
        # Detect pot region
        pot_region = self._detect_pot_region(pot_dets, table_bbox, board_region, w, h)
        
        # Detect player seats
        seats = self._detect_player_seats(player_dets, table_bbox, w, h)
        
        # Calculate overall confidence
        confidence = self._calculate_layout_confidence(
            table_bbox, board_region, pot_region, seats
        )
        
        # Create new layout
        new_layout = TableLayout(
            table_bbox=table_bbox,
            board_region=board_region,
            pot_region=pot_region,
            seats=seats,
            dealer_button_region=self._detect_button_region(button_dets, w, h),
            confidence=confidence,
            timestamp=time.time(),
            layout_version=(self.current_layout.layout_version + 1) if self.current_layout else 1,
            table_shape=self._estimate_table_shape(table_dets),
        )
        
        # Validate against history
        if self.layout_history and not force:
            prev_layout = self.layout_history[-1]
            iou = self._layout_iou(prev_layout, new_layout)
            
            if iou < self.LAYOUT_CHANGE_IOU_THRESHOLD:
                logger.warning(f"Large layout change detected (IoU={iou:.2f})")
                # Blend with previous layout for stability
                new_layout = self._blend_layouts(prev_layout, new_layout, alpha=0.3)
        
        # Update state
        self.previous_layout = self.current_layout
        self.current_layout = new_layout
        self.layout_history.append(new_layout)
        self.calibration_count += 1
        self.last_recalibration_frame = self.frames_since_calibration
        self.frames_since_calibration = 0
        
        # Track confidence
        self.confidence_history.append(confidence)
        
        logger.info(
            f"Calibration complete: confidence={confidence:.2f}, "
            f"{len(seats)} seats detected, version={new_layout.layout_version}"
        )
        
        return new_layout
    
    def _should_recalibrate(self) -> bool:
        """Determine if recalibration should be performed."""
        # Force periodic recalibration
        if self.frames_since_calibration >= self.RECALIBRATION_INTERVAL:
            return True
        
        # Check confidence drop
        if self.confidence_history:
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            if avg_confidence < self.CONFIDENCE_DROP_THRESHOLD:
                logger.info(f"Low confidence detected ({avg_confidence:.2f}), triggering recalibration")
                return True
        
        # No layout yet
        if self.current_layout is None:
            return True
        
        return False
    
    def _detect_table_boundary(
        self,
        detections: List[Any],
        width: int,
        height: int,
    ) -> Optional[NormalizedBox]:
        """Detect table boundary from detections or image analysis."""
        if detections:
            # Use highest confidence table detection
            best_det = max(detections, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best_det.bbox
            return NormalizedBox(
                x1 / width, y1 / height,
                x2 / width, y2 / height,
                confidence=best_det.confidence,
            )
        
        # Fallback: detect from image analysis
        # This would typically use edge detection and shape analysis
        # For now, return None to use template fallback
        return None
    
    def _detect_board_region(
        self,
        table_bbox: NormalizedBox,
        detections: List[Any],
        width: int,
        height: int,
    ) -> NormalizedBox:
        """Detect community card board region."""
        # Look for card detections in the center of the table
        card_dets = [d for d in detections if d.class_name == "card"]
        
        if card_dets:
            # Find cards near table center
            cx, cy = table_bbox.center
            center_cards = [
                d for d in card_dets
                if abs((d.bbox[0] + d.bbox[2]) / (2 * width) - cx) < 0.2
                and abs((d.bbox[1] + d.bbox[3]) / (2 * height) - cy) < 0.15
            ]
            
            if center_cards:
                x1 = min(d.bbox[0] for d in center_cards) / width
                y1 = min(d.bbox[1] for d in center_cards) / height
                x2 = max(d.bbox[2] for d in center_cards) / width
                y2 = max(d.bbox[3] for d in center_cards) / height
                
                # Add padding
                padding = 0.02
                return NormalizedBox(
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(1, x2 + padding),
                    min(1, y2 + padding),
                    confidence=0.8,
                )
        
        # Default: center of table
        tx1, ty1, tx2, ty2 = table_bbox.x1, table_bbox.y1, table_bbox.x2, table_bbox.y2
        tw, th = tx2 - tx1, ty2 - ty1
        
        return NormalizedBox(
            tx1 + tw * 0.35,
            ty1 + th * 0.35,
            tx1 + tw * 0.65,
            ty1 + th * 0.55,
            confidence=0.6,
        )
    
    def _detect_pot_region(
        self,
        pot_detections: List[Any],
        table_bbox: NormalizedBox,
        board_region: NormalizedBox,
        width: int,
        height: int,
    ) -> NormalizedBox:
        """Detect pot display region."""
        if pot_detections:
            best_det = max(pot_detections, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best_det.bbox
            return NormalizedBox(
                x1 / width, y1 / height,
                x2 / width, y2 / height,
                confidence=best_det.confidence,
            )
        
        # Default: below board region
        bx1, by1, bx2, by2 = board_region.x1, board_region.y1, board_region.x2, board_region.y2
        bw, bh = bx2 - bx1, by2 - by1
        
        return NormalizedBox(
            bx1 + bw * 0.2,
            by2,
            bx2 - bw * 0.2,
            min(1, by2 + bh * 0.5),
            confidence=0.5,
        )
    
    def _detect_player_seats(
        self,
        player_detections: List[Any],
        table_bbox: NormalizedBox,
        width: int,
        height: int,
    ) -> List[SeatInfo]:
        """Detect player seat positions."""
        seats = []
        
        if player_detections:
            # Sort players by angle around table center
            cx, cy = table_bbox.center
            cx_px, cy_px = cx * width, cy * height
            
            player_angles = []
            for det in player_detections:
                px = (det.bbox[0] + det.bbox[2]) / 2
                py = (det.bbox[1] + det.bbox[3]) / 2
                
                angle = math.atan2(py - cy_px, px - cx_px)
                player_angles.append((angle, det))
            
            player_angles.sort(key=lambda x: x[0])
            
            # Assign positions based on angle
            position_order = [
                SeatPosition.HERO,
                SeatPosition.SB,
                SeatPosition.BB,
                SeatPosition.UTG,
                SeatPosition.MP1,
                SeatPosition.MP2,
                SeatPosition.HJ,
                SeatPosition.CO,
                SeatPosition.BTN,
            ]
            
            for i, (angle, det) in enumerate(player_angles):
                if i >= len(position_order):
                    break
                
                x1, y1, x2, y2 = det.bbox
                
                # Expand bbox to create seat region
                pad_x = (x2 - x1) * 0.3
                pad_y = (y2 - y1) * 0.3
                
                seat_region = NormalizedBox(
                    max(0, (x1 - pad_x) / width),
                    max(0, (y1 - pad_y) / height),
                    min(1, (x2 + pad_x) / width),
                    min(1, (y2 + pad_y) / height),
                    confidence=det.confidence,
                )
                
                seats.append(SeatInfo(
                    position=position_order[i],
                    region=seat_region,
                    confidence=det.confidence,
                    is_active=True,
                ))
        else:
            # Use template-based seat positions
            template = self.LAYOUT_TEMPLATES.get(self.default_layout, {})
            seat_templates = template.get("seats", [])
            
            for pos, (sx1, sy1, sx2, sy2) in seat_templates:
                seats.append(SeatInfo(
                    position=pos,
                    region=NormalizedBox(sx1, sy1, sx2, sy2, confidence=0.5),
                    confidence=0.5,
                    is_active=True,
                ))
        
        return seats
    
    def _detect_button_region(
        self,
        button_detections: List[Any],
        width: int,
        height: int,
    ) -> Optional[NormalizedBox]:
        """Detect dealer button region."""
        if button_detections:
            best_det = max(button_detections, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best_det.bbox
            return NormalizedBox(
                x1 / width, y1 / height,
                x2 / width, y2 / height,
                confidence=best_det.confidence,
            )
        return None
    
    def _estimate_table_shape(self, detections: List[Any]) -> str:
        """Estimate table shape from detections."""
        if not detections:
            return "oval"
        
        # Analyze aspect ratios and corner detection
        # Simplified for now
        return "oval"
    
    def _calculate_layout_confidence(
        self,
        table_bbox: NormalizedBox,
        board_region: NormalizedBox,
        pot_region: NormalizedBox,
        seats: List[SeatInfo],
    ) -> float:
        """Calculate overall layout confidence score."""
        scores = []
        
        # Table bbox confidence
        scores.append(table_bbox.confidence)
        
        # Board region confidence
        scores.append(board_region.confidence)
        
        # Pot region confidence
        scores.append(pot_region.confidence)
        
        # Seat confidences
        if seats:
            seat_conf = sum(s.confidence for s in seats) / len(seats)
            scores.append(seat_conf)
        
        # Bonus for having all components
        completeness = len([s for s in scores if s > 0]) / 4
        scores.append(completeness)
        
        return sum(scores) / len(scores)
    
    def _layout_iou(self, layout1: TableLayout, layout2: TableLayout) -> float:
        """Calculate IoU between two layouts."""
        ious = []
        
        # Table bbox IoU
        ious.append(layout1.table_bbox.iou(layout2.table_bbox))
        
        # Board region IoU
        ious.append(layout1.board_region.iou(layout2.board_region))
        
        # Pot region IoU
        ious.append(layout1.pot_region.iou(layout2.pot_region))
        
        if not ious:
            return 1.0
        
        return sum(ious) / len(ious)
    
    def _blend_layouts(
        self,
        layout1: TableLayout,
        layout2: TableLayout,
        alpha: float = 0.5,
    ) -> TableLayout:
        """Blend two layouts together for smooth transitions."""
        beta = 1 - alpha
        
        # Blend table bbox
        t1, t2 = layout1.table_bbox, layout2.table_bbox
        blended_table = NormalizedBox(
            alpha * t2.x1 + beta * t1.x1,
            alpha * t2.y1 + beta * t1.y1,
            alpha * t2.x2 + beta * t1.x2,
            alpha * t2.y2 + beta * t1.y2,
            confidence=alpha * t2.confidence + beta * t1.confidence,
        )
        
        # Blend board region
        b1, b2 = layout1.board_region, layout2.board_region
        blended_board = NormalizedBox(
            alpha * b2.x1 + beta * b1.x1,
            alpha * b2.y1 + beta * b1.y1,
            alpha * b2.x2 + beta * b1.x2,
            alpha * b2.y2 + beta * b1.y2,
            confidence=alpha * b2.confidence + beta * b1.confidence,
        )
        
        # Blend pot region
        p1, p2 = layout1.pot_region, layout2.pot_region
        blended_pot = NormalizedBox(
            alpha * p2.x1 + beta * p1.x1,
            alpha * p2.y1 + beta * p1.y1,
            alpha * p2.x2 + beta * p1.x2,
            alpha * p2.y2 + beta * p1.y2,
            confidence=alpha * p2.confidence + beta * p1.confidence,
        )
        
        # Blend seats (match by position)
        blended_seats = []
        for seat2 in layout2.seats:
            seat1 = layout1.get_seat_by_position(seat2.position)
            if seat1:
                s1, s2 = seat1.region, seat2.region
                blended_region = NormalizedBox(
                    alpha * s2.x1 + beta * s1.x1,
                    alpha * s2.y1 + beta * s1.y1,
                    alpha * s2.x2 + beta * s1.x2,
                    alpha * s2.y2 + beta * s1.y2,
                    confidence=alpha * s2.confidence + beta * s1.confidence,
                )
                blended_seats.append(SeatInfo(
                    position=seat2.position,
                    region=blended_region,
                    confidence=alpha * seat2.confidence + beta * seat1.confidence,
                    is_active=seat2.is_active or seat1.is_active,
                ))
            else:
                blended_seats.append(seat2)
        
        return TableLayout(
            table_bbox=blended_table,
            board_region=blended_board,
            pot_region=blended_pot,
            seats=blended_seats,
            confidence=alpha * layout2.confidence + beta * layout1.confidence,
            timestamp=time.time(),
            layout_version=layout2.layout_version,
            table_shape=layout2.table_shape if alpha > 0.5 else layout1.table_shape,
        )
    
    def normalize_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> NormalizedBox:
        """Convert pixel bbox to normalized coordinates."""
        return NormalizedBox(
            bbox[0] / image_width,
            bbox[1] / image_height,
            bbox[2] / image_width,
            bbox[3] / image_height,
        )
    
    def denormalize_bbox(
        self,
        norm_box: NormalizedBox,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """Convert normalized box to pixel coordinates."""
        return norm_box.to_pixel_coords(image_width, image_height)
    
    def get_roi_for_class(
        self,
        class_name: str,
        image_width: int,
        image_height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get adaptive ROI for a specific class based on table layout.
        
        Args:
            class_name: Object class name ("card", "chip", "player", etc.)
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Pixel coordinates (x1, y1, x2, y2) or None
        """
        if not self.current_layout:
            return None
        
        layout = self.current_layout
        
        if class_name == "card":
            # Cards can appear on board or in player hands
            # Return combined region
            regions = [layout.board_region]
            for seat in layout.seats:
                if seat.card_region:
                    regions.append(seat.card_region)
            
            # Compute bounding box of all regions
            return self._merge_regions(regions, image_width, image_height)
        
        elif class_name == "chip":
            # Chips appear near players and in pot area
            regions = [layout.pot_region]
            for seat in layout.seats:
                if seat.chip_region:
                    regions.append(seat.chip_region)
                else:
                    # Use seat region as fallback
                    regions.append(seat.region)
            
            return self._merge_regions(regions, image_width, image_height)
        
        elif class_name == "player":
            # Players are at seat positions
            regions = [s.region for s in layout.seats]
            return self._merge_regions(regions, image_width, image_height)
        
        elif class_name == "pot":
            return layout.pot_region.to_pixel_coords(image_width, image_height)
        
        elif class_name == "dealer_button":
            if layout.dealer_button_region:
                return layout.dealer_button_region.to_pixel_coords(image_width, image_height)
        
        elif class_name == "table":
            return layout.table_bbox.to_pixel_coords(image_width, image_height)
        
        return None
    
    def _merge_regions(
        self,
        regions: List[NormalizedBox],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Merge multiple regions into a single bounding box."""
        if not regions:
            return None
        
        x1 = min(r.x1 for r in regions)
        y1 = min(r.y1 for r in regions)
        x2 = max(r.x2 for r in regions)
        y2 = max(r.y2 for r in regions)
        
        return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return {
            "calibration_count": self.calibration_count,
            "frames_since_calibration": self.frames_since_calibration,
            "current_confidence": self.current_layout.confidence if self.current_layout else 0.0,
            "avg_confidence": (
                sum(self.confidence_history) / len(self.confidence_history)
                if self.confidence_history else 0.0
            ),
            "layout_version": self.current_layout.layout_version if self.current_layout else 0,
            "num_seats": len(self.current_layout.seats) if self.current_layout else 0,
        }
    
    def reset(self) -> None:
        """Reset calibration state."""
        self.current_layout = None
        self.layout_history.clear()
        self.detection_history.clear()
        self.confidence_history.clear()
        self.calibration_count = 0
        self.frames_since_calibration = 0
        logger.info("Table calibration reset")
