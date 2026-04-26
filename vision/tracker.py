"""
vision/tracker.py - Temporal Tracking Module

Critical component for stable predictions across frames:
- Object persistence (objects don't disappear between frames)
- Temporal smoothing (predictions don't flicker)
- Motion prediction (Kalman filtering)
- Data association (matching detections to tracked objects)
- Confidence decay (old tracks lose confidence over time)

Implements custom temporal tracker optimized for poker table objects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Deque
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """
    Represents a tracked object across frames.
    
    Attributes:
        track_id: Unique identifier for this track
        class_id: Object class ID
        class_name: Object class name
        bbox: Current bounding box [x1, y1, x2, y2]
        confidence: Current detection confidence
        state_history: History of states for temporal analysis
        created_at: Timestamp when track was created
        updated_at: Timestamp of last update
        frames_seen: Number of frames this object has been seen
        frames_missed: Consecutive frames without detection
        classification: Optional card classification result
    """
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    state_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    frames_seen: int = 1
    frames_missed: int = 0
    classification: Optional[Any] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Current center position."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> int:
        """Current bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Estimated velocity based on history."""
        if len(self.state_history) < 2:
            return (0.0, 0.0)
        
        oldest = self.state_history[0]
        newest = self.state_history[-1]
        
        dt = newest.get("timestamp", 0) - oldest.get("timestamp", 0)
        if dt <= 0:
            return (0.0, 0.0)
        
        dx = newest["center"][0] - oldest["center"][0]
        dy = newest["center"][1] - oldest["center"][1]
        
        return (dx / dt, dy / dt)
    
    @property
    def is_stable(self) -> bool:
        """Check if track is stable (seen multiple frames)."""
        return self.frames_seen >= 3
    
    @property
    def temporal_confidence(self) -> float:
        """Confidence boosted by temporal consistency."""
        base_conf = self.confidence
        
        # Boost for stability
        stability_boost = min(0.2, self.frames_seen * 0.02)
        
        # Penalty for missed frames
        miss_penalty = self.frames_missed * 0.1
        
        return max(0.0, min(1.0, base_conf + stability_boost - miss_penalty))
    
    def predict_position(self, dt: float = 0.033) -> Tuple[float, float]:
        """Predict position dt seconds in the future."""
        cx, cy = self.center
        vx, vy = self.velocity
        
        return (cx + vx * dt, cy + vy * dt)
    
    def record_state(self, bbox: Tuple[int, int, int, int], confidence: float, timestamp: float):
        """Record current state to history."""
        self.state_history.append({
            "bbox": bbox,
            "confidence": confidence,
            "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            "timestamp": timestamp,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "temporal_confidence": self.temporal_confidence,
            "center": self.center,
            "velocity": self.velocity,
            "is_stable": self.is_stable,
            "frames_seen": self.frames_seen,
            "frames_missed": self.frames_missed,
            "classification": self.classification.to_dict() if self.classification else None,
        }


class TrackerState(Enum):
    """Tracker state machine states."""
    DETECTED = 0      # Just detected
    TRACKING = 1      # Actively tracking
    UNCERTAIN = 2     # Missed recently, may disappear
    LOST = 3          # Track lost


class Tracker:
    """
    Custom temporal tracker for poker table objects.
    
    Features:
    - Kalman-like motion prediction
    - IoU-based data association
    - Track management (creation, deletion)
    - Temporal smoothing
    - Confidence decay
    """
    
    # Configuration
    MATCH_IOU_THRESHOLD = 0.3       # Minimum IoU for matching
    MATCH_DISTANCE_THRESHOLD = 100  # Maximum center distance for matching
    CONFIRM_THRESHOLD = 3           # Frames to confirm track
    MAX_AGE = 30                    # Max frames without detection before deletion
    SMOOTHING_ALPHA = 0.3           # Exponential smoothing factor
    
    def __init__(
        self,
        max_tracks: int = 100,
        confirm_threshold: int = CONFIRM_THRESHOLD,
        max_age: int = MAX_AGE,
        iou_threshold: float = MATCH_IOU_THRESHOLD,
    ):
        """
        Initialize tracker.
        
        Args:
            max_tracks: Maximum number of simultaneous tracks
            confirm_threshold: Frames needed to confirm a track
            max_age: Max frames without detection before deletion
            iou_threshold: IoU threshold for data association
        """
        self.max_tracks = max_tracks
        self.confirm_threshold = confirm_threshold
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        
        # Track storage
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        
        # Frame counter
        self.frame_count = 0
        self.last_timestamp = 0.0
        
        logger.info(f"Tracker initialized (max_tracks={max_tracks})")
    
    def update(
        self,
        detections: List[Any],
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of DetectionResult from YOLO
            timestamp: Frame timestamp (auto-generated if None)
            
        Returns:
            List of updated TrackedObject
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        self.last_timestamp = timestamp
        
        # Increment missed frame counters
        for track in self.tracks.values():
            track.frames_missed += 1
        
        # Data association
        matched_detections = set()
        
        for track_id, track in list(self.tracks.items()):
            if track.frames_missed > self.max_age:
                # Remove old tracks
                logger.debug(f"Removing track {track_id} (age={track.frames_missed})")
                del self.tracks[track_id]
                continue
            
            # Find best matching detection
            best_match_idx = None
            best_iou = 0.0
            
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                
                # Check class match
                if det.class_id != track.class_id:
                    continue
                
                # Calculate IoU
                iou = self._compute_iou(track.bbox, det.bbox)
                
                # Also check center distance
                dist = self._euclidean_distance(track.center, det.center)
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match_idx = idx
                elif dist < self.MATCH_DISTANCE_THRESHOLD and iou > best_iou * 0.5:
                    # Fallback to distance-based matching
                    best_iou = iou * 0.5
                    best_match_idx = idx
            
            if best_match_idx is not None:
                # Update track with matched detection
                det = detections[best_match_idx]
                self._update_track(track, det, timestamp)
                matched_detections.add(best_match_idx)
            else:
                # Apply exponential smoothing to bbox
                track.bbox = self._smooth_bbox(track)
        
        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                self._create_track(det, timestamp)
        
        # Return active tracks sorted by confidence
        active_tracks = [
            t for t in self.tracks.values()
            if t.frames_missed <= self.max_age // 2
        ]
        active_tracks.sort(key=lambda t: t.temporal_confidence, reverse=True)
        
        return active_tracks
    
    def _create_track(self, detection: Any, timestamp: float) -> None:
        """Create a new track from detection."""
        if len(self.tracks) >= self.max_tracks:
            # Remove lowest confidence track
            min_track = min(self.tracks.values(), key=lambda t: t.temporal_confidence)
            if min_track.temporal_confidence < detection.confidence:
                del self.tracks[min_track.track_id]
            else:
                return  # Don't create new track
        
        track_id = self.next_track_id
        self.next_track_id += 1
        
        track = TrackedObject(
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence,
            created_at=timestamp,
            updated_at=timestamp,
        )
        track.record_state(detection.bbox, detection.confidence, timestamp)
        
        self.tracks[track_id] = track
        logger.debug(f"Created track {track_id} for {detection.class_name}")
    
    def _update_track(
        self,
        track: TrackedObject,
        detection: Any,
        timestamp: float,
    ) -> None:
        """Update existing track with new detection."""
        # Exponential smoothing for bbox
        alpha = self.SMOOTHING_ALPHA
        x1, y1, x2, y2 = track.bbox
        dx1, dy1, dx2, dy2 = detection.bbox
        
        new_bbox = (
            int(alpha * dx1 + (1 - alpha) * x1),
            int(alpha * dy1 + (1 - alpha) * y1),
            int(alpha * dx2 + (1 - alpha) * x2),
            int(alpha * dy2 + (1 - alpha) * y2),
        )
        
        track.bbox = new_bbox
        track.confidence = alpha * detection.confidence + (1 - alpha) * track.confidence
        track.updated_at = timestamp
        track.frames_seen += 1
        track.frames_missed = 0
        
        # Record state
        track.record_state(new_bbox, track.confidence, timestamp)
        
        # Update classification if provided
        if hasattr(detection, 'classification') and detection.classification:
            track.classification = detection.classification
    
    def _smooth_bbox(self, track: TrackedObject) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing to bounding box."""
        if len(track.state_history) < 2:
            return track.bbox
        
        # Use velocity prediction
        predicted_cx, predicted_cy = track.predict_position()
        
        # Blend with current bbox
        cx, cy = track.center
        w = track.bbox[2] - track.bbox[0]
        h = track.bbox[3] - track.bbox[1]
        
        blend_alpha = 0.7
        smooth_cx = blend_alpha * cx + (1 - blend_alpha) * predicted_cx
        smooth_cy = blend_alpha * cy + (1 - blend_alpha) * predicted_cy
        
        return (
            int(smooth_cx - w / 2),
            int(smooth_cy - h / 2),
            int(smooth_cx + w / 2),
            int(smooth_cy + h / 2),
        )
    
    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Intersection
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = max(0, (x2 - x1) * (y2 - y1))
        area2 = max(0, (x4 - x3) * (y4 - y3))
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _euclidean_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
    ) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_tracks_by_class(self, class_id: int) -> List[TrackedObject]:
        """Get all tracks of a specific class."""
        return [t for t in self.tracks.values() if t.class_id == class_id]
    
    def get_stable_tracks(self) -> List[TrackedObject]:
        """Get only stable tracks."""
        return [t for t in self.tracks.values() if t.is_stable]
    
    def reset(self) -> None:
        """Reset tracker (clear all tracks)."""
        self.tracks.clear()
        self.frame_count = 0
        logger.info("Tracker reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "total_tracks": len(self.tracks),
            "stable_tracks": sum(1 for t in self.tracks.values() if t.is_stable),
            "frame_count": self.frame_count,
            "tracks_by_class": {},
        }
