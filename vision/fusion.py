"""
vision/fusion.py - Multi-Source Fusion Engine

Combines detections from multiple sources with weighted decision making:
- YOLO detection confidence
- CNN classification confidence
- Temporal consistency from tracker
- Rule-based validation

No single model decides alone - all signals are fused for robust decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusedDetection:
    """
    Result of fusion engine for a single object.
    
    Combines:
    - Detection confidence (YOLO)
    - Classification confidence (CNN)
    - Temporal confidence (Tracker)
    - Final fused confidence score
    """
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    
    # Source confidences
    detection_confidence: float = 0.0
    classification_confidence: float = 0.0
    temporal_confidence: float = 0.0
    
    # Fused result
    fused_confidence: float = 0.0
    is_reliable: bool = False
    
    # Card-specific fields
    rank: str = ""
    suit: str = ""
    
    # Metadata
    age_frames: int = 0
    source_count: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "detection_confidence": self.detection_confidence,
            "classification_confidence": self.classification_confidence,
            "temporal_confidence": self.temporal_confidence,
            "fused_confidence": self.fused_confidence,
            "is_reliable": self.is_reliable,
            "rank": self.rank,
            "suit": self.suit,
            "age_frames": self.age_frames,
            "source_count": self.source_count,
        }


class FusionWeight(Enum):
    """Weight categories for fusion."""
    DETECTION = 0.4      # YOLO confidence weight
    CLASSIFICATION = 0.35  # CNN confidence weight
    TEMPORAL = 0.25      # Tracker confidence weight


class FusionEngine:
    """
    Multi-source fusion engine for robust object identification.
    
    Features:
    - Weighted confidence fusion
    - Adaptive weighting based on conditions
    - Outlier rejection
    - Confidence calibration
    """
    
    # Default weights
    DEFAULT_WEIGHTS = {
        "detection": 0.40,
        "classification": 0.35,
        "temporal": 0.25,
    }
    
    # Confidence thresholds
    RELIABLE_THRESHOLD = 0.6
    MINIMUM_THRESHOLD = 0.3
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        reliable_threshold: float = RELIABLE_THRESHOLD,
        minimum_threshold: float = MINIMUM_THRESHOLD,
    ):
        """
        Initialize fusion engine.
        
        Args:
            weights: Custom weights for fusion
            reliable_threshold: Threshold for marking detection as reliable
            minimum_threshold: Minimum confidence to keep detection
        """
        self.weights = {**self.DEFAULT_WEIGHTS}
        if weights:
            self.weights.update(weights)
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        self.reliable_threshold = reliable_threshold
        self.minimum_threshold = minimum_threshold
        
        # Statistics
        self.fusion_count = 0
        self.rejection_count = 0
        
        logger.info(f"FusionEngine initialized with weights: {self.weights}")
    
    def fuse(
        self,
        tracked_objects: List[Any],
    ) -> List[FusedDetection]:
        """
        Fuse information from tracked objects.
        
        Args:
            tracked_objects: List of TrackedObject from tracker
            
        Returns:
            List of FusedDetection with combined confidence scores
        """
        fused_detections = []
        
        for track in tracked_objects:
            # Extract confidences from different sources
            detection_conf = getattr(track, 'confidence', 0.5)
            temporal_conf = getattr(track, 'temporal_confidence', 0.5)
            
            # Get classification confidence if available
            classification = getattr(track, 'classification', None)
            if classification:
                classification_conf = getattr(classification, 'combined_confidence', 0.5)
                rank = getattr(classification, 'rank', '')
                suit = getattr(classification, 'suit', '')
            else:
                classification_conf = 0.5  # Neutral when unavailable
                rank = ''
                suit = ''
            
            # Compute fused confidence
            fused_conf = self._compute_fused_confidence(
                detection_conf=detection_conf,
                classification_conf=classification_conf,
                temporal_conf=temporal_conf,
            )
            
            # Determine reliability
            is_reliable = (
                fused_conf >= self.reliable_threshold and
                detection_conf >= self.minimum_threshold and
                temporal_conf >= self.minimum_threshold
            )
            
            # Count active sources
            source_count = 0
            if detection_conf >= self.minimum_threshold:
                source_count += 1
            if classification_conf >= self.minimum_threshold:
                source_count += 1
            if temporal_conf >= self.minimum_threshold:
                source_count += 1
            
            fused_det = FusedDetection(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                bbox=track.bbox,
                detection_confidence=detection_conf,
                classification_confidence=classification_conf,
                temporal_confidence=temporal_conf,
                fused_confidence=fused_conf,
                is_reliable=is_reliable,
                rank=rank,
                suit=suit,
                age_frames=getattr(track, 'frames_seen', 0),
                source_count=source_count,
            )
            
            fused_detections.append(fused_det)
        
        self.fusion_count += len(fused_detections)
        
        # Sort by fused confidence
        fused_detections.sort(key=lambda x: x.fused_confidence, reverse=True)
        
        return fused_detections
    
    def _compute_fused_confidence(
        self,
        detection_conf: float,
        classification_conf: float,
        temporal_conf: float,
    ) -> float:
        """
        Compute weighted fusion of confidence scores.
        
        Uses adaptive weighting based on confidence levels.
        """
        # Base weighted average
        fused = (
            self.weights["detection"] * detection_conf +
            self.weights["classification"] * classification_conf +
            self.weights["temporal"] * temporal_conf
        )
        
        # Apply penalty for low agreement between sources
        confs = [detection_conf, classification_conf, temporal_conf]
        variance = np.var(confs)
        
        if variance > 0.1:
            # High disagreement - reduce confidence
            penalty = 0.1 * variance
            fused -= penalty
        
        # Apply bonus for high agreement
        if variance < 0.02 and min(confs) > 0.5:
            # All sources agree with good confidence
            bonus = 0.05
            fused += bonus
        
        return max(0.0, min(1.0, fused))
    
    def fuse_cards(
        self,
        card_tracks: List[Any],
        classifications: List[Any],
    ) -> List[FusedDetection]:
        """
        Specialized fusion for card detections with classifications.
        
        Args:
            card_tracks: List of card TrackedObject
            classifications: List of CardClassificationResult
            
        Returns:
            List of FusedDetection for cards
        """
        if not card_tracks:
            return []
        
        # Match tracks to classifications by position or index
        fused_detections = []
        
        for i, track in enumerate(card_tracks):
            # Find best matching classification
            best_class = None
            best_class_conf = 0.0
            
            if classifications and i < len(classifications):
                cls = classifications[i]
                best_class = cls
                best_class_conf = getattr(cls, 'combined_confidence', 0.5)
            
            detection_conf = getattr(track, 'confidence', 0.5)
            temporal_conf = getattr(track, 'temporal_confidence', 0.5)
            
            # For cards, increase classification weight if we have a confident classification
            if best_class_conf > 0.7:
                weights = {
                    "detection": 0.25,
                    "classification": 0.50,
                    "temporal": 0.25,
                }
            else:
                weights = self.weights
            
            fused_conf = (
                weights["detection"] * detection_conf +
                weights["classification"] * best_class_conf +
                weights["temporal"] * temporal_conf
            )
            
            is_reliable = (
                fused_conf >= self.reliable_threshold and
                detection_conf >= self.minimum_threshold and
                temporal_conf >= self.minimum_threshold
            )
            
            rank = getattr(best_class, 'rank', '') if best_class else ''
            suit = getattr(best_class, 'suit', '') if best_class else ''
            
            fused_det = FusedDetection(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name="card",
                bbox=track.bbox,
                detection_confidence=detection_conf,
                classification_confidence=best_class_conf,
                temporal_confidence=temporal_conf,
                fused_confidence=fused_conf,
                is_reliable=is_reliable,
                rank=rank,
                suit=suit,
                age_frames=getattr(track, 'frames_seen', 0),
                source_count=3 if best_class else 2,
            )
            
            fused_detections.append(fused_det)
        
        return fused_detections
    
    def filter_unreliable(
        self,
        fused_detections: List[FusedDetection],
        min_confidence: Optional[float] = None,
    ) -> List[FusedDetection]:
        """
        Filter out unreliable detections.
        
        Args:
            fused_detections: List of FusedDetection
            min_confidence: Override minimum confidence threshold
            
        Returns:
            Filtered list of reliable detections
        """
        threshold = min_confidence or self.minimum_threshold
        
        filtered = [
            det for det in fused_detections
            if det.fused_confidence >= threshold
        ]
        
        rejected = len(fused_detections) - len(filtered)
        self.rejection_count += rejected
        
        if rejected > 0:
            logger.debug(f"Filtered {rejected} unreliable detections")
        
        return filtered
    
    def get_top_detections(
        self,
        fused_detections: List[FusedDetection],
        class_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[FusedDetection]:
        """
        Get top K detections, optionally filtered by class.
        
        Args:
            fused_detections: List of FusedDetection
            class_name: Filter by class name
            top_k: Number of top detections to return
            
        Returns:
            Top K detections
        """
        if class_name:
            filtered = [d for d in fused_detections if d.class_name == class_name]
        else:
            filtered = fused_detections
        
        return sorted(filtered, key=lambda x: x.fused_confidence, reverse=True)[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        return {
            "total_fusions": self.fusion_count,
            "total_rejections": self.rejection_count,
            "weights": self.weights,
            "reliable_threshold": self.reliable_threshold,
            "minimum_threshold": self.minimum_threshold,
        }
