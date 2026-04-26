"""
vision/pipeline.py - Main Poker Vision Pipeline

Production-grade async pipeline integrating all components:
- YOLO detection
- CNN card classification
- Temporal tracking
- Multi-source fusion
- State machine with validation
- Guardrails for error prevention
- Dynamic table calibration

Features:
- Real-time processing (>= 20 FPS target)
- Async operation with separate threads
- Frame skipping under load
- Comprehensive logging and metrics
- Confidence visualization
- Resolution-independent coordinate system
- Auto-recalibration support
"""

from __future__ import annotations

import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import cv2
import numpy as np

from .yolo_detector import YOLODetector, DetectionResult
from .card_classifier import CardClassifier, CardClassificationResult
from .tracker import Tracker, TrackedObject
from .fusion import FusionEngine, FusedDetection
from .state_machine import StateMachine, PokerTableState, Street
from .guardrails import Guardrails, ValidationResult, ValidationStatus
from .table_calibration import TableCalibrator, TableLayout

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the vision pipeline."""
    
    # Model paths
    yolo_model_path: Optional[str] = None
    classifier_model_path: Optional[str] = None
    
    # Device settings
    device: str = "auto"  # "cuda", "cpu", "auto"
    
    # Performance
    max_fps: float = 30.0
    enable_async: bool = True
    frame_skip_threshold: float = 0.8  # Skip frames if processing takes >80% of frame time
    
    # Confidence thresholds
    min_detection_confidence: float = 0.4
    min_classification_confidence: float = 0.4
    min_fused_confidence: float = 0.5
    
    # Tracking
    max_tracks: int = 100
    track_confirm_frames: int = 3
    
    # State machine
    enable_rollback: bool = True
    max_players: int = 9
    
    # Guardrails
    enable_guardrails: bool = True
    
    # Table calibration
    enable_calibration: bool = True
    default_layout: str = "6max_standard"
    calibration_interval: int = 300  # Frames between recalibrations
    
    # Debug
    debug_mode: bool = False
    visualize: bool = False
    
    # Mock mode for testing without models
    use_mock: bool = False


@dataclass
class PipelineMetrics:
    """Performance metrics for the pipeline."""
    
    frames_processed: int = 0
    frames_dropped: int = 0
    total_processing_time: float = 0.0
    detection_time: float = 0.0
    classification_time: float = 0.0
    tracking_time: float = 0.0
    fusion_time: float = 0.0
    state_update_time: float = 0.0
    
    fps_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    @property
    def current_fps(self) -> float:
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    @property
    def avg_processing_time(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_processing_time / self.frames_processed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "current_fps": self.current_fps,
            "avg_processing_time_ms": self.avg_processing_time * 1000,
            "detection_time_ms": self.detection_time * 1000,
            "classification_time_ms": self.classification_time * 1000,
            "tracking_time_ms": self.tracking_time * 1000,
            "fusion_time_ms": self.fusion_time * 1000,
            "state_update_time_ms": self.state_update_time * 1000,
        }


class PokerVisionPipeline:
    """
    Production-grade poker vision pipeline.
    
    Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │   Camera    │ ──→ │   YOLOv8     │ ──→ │  Classifier │
    │   Capture   │     │  Detection   │     │   (Cards)   │
    └─────────────┘     └──────────────┘     └─────────────┘
                              │                    │
                              ▼                    ▼
                        ┌──────────────┐     ┌─────────────┐
                        │   Tracker    │ ←── │   Fusion    │
                        │  (Temporal)  │     │   Engine    │
                        └──────────────┘     └─────────────┘
                              │
                              ▼
                        ┌──────────────┐     ┌─────────────┐
                        │   State      │ ←── │  Guardrails │
                        │   Machine    │     │  Validation │
                        └──────────────┘     └─────────────┘
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_state_update: Optional[Callable[[PokerTableState], None]] = None,
    ):
        """
        Initialize the vision pipeline.
        
        Args:
            config: Pipeline configuration
            on_state_update: Callback for state updates
        """
        self.config = config or PipelineConfig()
        self.on_state_update = on_state_update
        
        # Initialize components
        self.detector = YOLODetector(
            model_path=self.config.yolo_model_path,
            device=self.config.device,
            use_mock=self.config.use_mock,
        )
        
        self.classifier = CardClassifier(
            model_path=self.config.classifier_model_path,
            device=self.config.device,
            use_mock=self.config.use_mock,
        )
        
        self.tracker = Tracker(
            max_tracks=self.config.max_tracks,
            confirm_threshold=self.config.track_confirm_frames,
        )
        
        self.fusion = FusionEngine(
            minimum_threshold=self.config.min_fused_confidence,
        )
        
        self.state_machine = StateMachine(
            max_players=self.config.max_players,
            enable_rollback=self.config.enable_rollback,
        )
        
        self.guardrails = Guardrails(
            enable_logging=self.config.debug_mode,
        )
        
        # Table calibration (NEW)
        self.calibrator = TableCalibrator(
            default_layout=self.config.default_layout,
            enable_auto_detect=self.config.enable_calibration,
        )
        
        # State
        self.is_running = False
        self.frame_number = 0
        self.current_state = PokerTableState()
        self.previous_state: Optional[PokerTableState] = None
        self.current_image_shape: Optional[Tuple[int, int]] = None
        
        # Async processing
        self.executor: Optional[ThreadPoolExecutor] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self.result_queue: queue.Queue = queue.Queue()
        
        # Metrics
        self.metrics = PipelineMetrics()
        
        # Visualization
        self.visualization_callback: Optional[Callable] = None
        
        logger.info("PokerVisionPipeline initialized")
    
    def process_frame(
        self,
        image: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> PokerTableState:
        """
        Process a single frame through the pipeline.
        
        Args:
            image: Input frame (BGR format)
            timestamp: Frame timestamp (auto-generated if None)
            
        Returns:
            Updated PokerTableState
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = start_time
        
        self.frame_number += 1
        
        try:
            # Store image shape for calibration
            self.current_image_shape = image.shape[:2]
            
            # ── Stage 0: Table Calibration (if enabled) ──────────────────
            if self.config.enable_calibration:
                # Update calibration periodically or on first frame
                if self.frame_number == 1 or self.frame_number % self.config.calibration_interval == 0:
                    self.calibrator.calibrate_from_detections(
                        detections=[],  # Will be populated below
                        image_shape=image.shape[:2],
                        force=(self.frame_number == 1),
                    )
            
            # ── Stage 1: Detection ──────────────────────────────────────
            det_start = time.time()
            detections = self.detector.detect(image)
            
            # Update calibration with actual detections
            if self.config.enable_calibration and self.frame_number % self.config.calibration_interval == 0:
                self.calibrator.calibrate_from_detections(
                    detections=detections,
                    image_shape=image.shape[:2],
                    force=False,
                )
            
            # Set timestamps on detections
            for det in detections:
                det.timestamp = timestamp
            
            self.metrics.detection_time = time.time() - det_start
            
            # ── Stage 2: Card Classification ────────────────────────────
            cls_start = time.time()
            card_detections = [d for d in detections if d.class_name == 'card']
            
            classifications = []
            if card_detections:
                for det in card_detections:
                    x1, y1, x2, y2 = det.bbox
                    card_roi = image[y1:y2, x1:x2]
                    
                    classification = self.classifier.classify(card_roi, bbox=det.bbox)
                    classifications.append(classification)
                    
                    # Attach classification to detection
                    det.classification = classification
            
            self.metrics.classification_time = time.time() - cls_start
            
            # ── Stage 3: Tracking ───────────────────────────────────────
            track_start = time.time()
            tracked_objects = self.tracker.update(detections, timestamp=timestamp)
            self.metrics.tracking_time = time.time() - track_start
            
            # ── Stage 4: Fusion ─────────────────────────────────────────
            fusion_start = time.time()
            fused_detections = self.fusion.fuse(tracked_objects)
            self.metrics.fusion_time = time.time() - fusion_start
            
            # Filter unreliable detections
            reliable_detections = self.fusion.filter_unreliable(fused_detections)
            
            # ── Stage 5: State Machine Update ───────────────────────────
            state_start = time.time()
            
            if self.config.enable_guardrails and self.previous_state:
                # Validate before updating state
                validation = self.guardrails.validate_frame(
                    fused_detections=reliable_detections,
                    current_state=self.current_state,
                    previous_state=self.previous_state,
                    frame_number=self.frame_number,
                )
                
                if validation.should_rollback:
                    logger.warning("Guardrails triggered rollback")
                    self.state_machine.rollback()
            
            new_state = self.state_machine.update(
                fused_detections=reliable_detections,
                frame_number=self.frame_number,
            )
            
            self.metrics.state_update_time = time.time() - state_start
            
            # ── Update State ────────────────────────────────────────────
            self.previous_state = self.current_state
            self.current_state = new_state
            
            # Callback
            if self.on_state_update:
                self.on_state_update(self.current_state)
            
            # ── Metrics ─────────────────────────────────────────────────
            total_time = time.time() - start_time
            self.metrics.frames_processed += 1
            self.metrics.total_processing_time += total_time
            
            fps = 1.0 / total_time if total_time > 0 else 0
            self.metrics.fps_history.append(fps)
            
            # Frame skip decision
            frame_budget = 1.0 / self.config.max_fps
            if total_time > frame_budget * self.config.frame_skip_threshold:
                logger.debug(f"Frame processing slow: {total_time*1000:.1f}ms")
            
            if self.config.debug_mode:
                logger.debug(
                    f"Frame {self.frame_number}: {len(detections)} detections, "
                    f"{len(card_detections)} cards, "
                    f"processing time: {total_time*1000:.1f}ms, "
                    f"FPS: {fps:.1f}"
                )
            
            return self.current_state
            
        except Exception as e:
            logger.error(f"Pipeline error on frame {self.frame_number}: {e}")
            return self.current_state
    
    def process_frame_async(
        self,
        image: np.ndarray,
        callback: Optional[Callable[[PokerTableState], None]] = None,
    ) -> bool:
        """
        Submit frame for async processing.
        
        Args:
            image: Input frame
            callback: Optional callback for result
            
        Returns:
            True if frame was queued successfully
        """
        if not self.is_running:
            return False
        
        try:
            self.frame_queue.put_nowait((image, callback))
            return True
        except queue.Full:
            self.metrics.frames_dropped += 1
            logger.debug("Frame queue full, dropping frame")
            return False
    
    def _async_worker(self) -> None:
        """Async worker thread for processing frames."""
        while self.is_running:
            try:
                image, callback = self.frame_queue.get(timeout=0.1)
                result = self.process_frame(image)
                
                if callback:
                    callback(result)
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Async worker error: {e}")
    
    def start(self) -> None:
        """Start the async pipeline."""
        if self.is_running:
            return
        
        self.is_running = True
        
        if self.config.enable_async:
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.executor.submit(self._async_worker)
        
        logger.info("Pipeline started")
    
    def stop(self) -> None:
        """Stop the async pipeline."""
        self.is_running = False
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Pipeline stopped")
    
    def get_current_state(self) -> PokerTableState:
        """Get the current table state."""
        return self.current_state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return {
            "pipeline": self.metrics.to_dict(),
            "detector": self.detector.get_model_info(),
            "classifier": self.classifier.get_model_info(),
            "tracker": self.tracker.get_stats(),
            "fusion": self.fusion.get_stats(),
            "state_machine": self.state_machine.get_stats(),
            "guardrails": self.guardrails.get_stats(),
            "calibration": self.calibrator.get_stats(),
        }
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.tracker.reset()
        self.state_machine.reset()
        self.guardrails.reset()
        self.calibrator.reset()
        self.frame_number = 0
        self.metrics = PipelineMetrics()
        logger.info("Pipeline reset")
    
    def get_table_layout(self) -> Optional[TableLayout]:
        """Get current table layout from calibrator."""
        return self.calibrator.current_layout
    
    def force_recalibrate(self) -> TableLayout:
        """Force immediate table recalibration."""
        if self.current_image_shape:
            return self.calibrator.initialize_from_template(self.config.default_layout)
        raise RuntimeError("No image processed yet, cannot calibrate")
    
    def visualize(
        self,
        image: np.ndarray,
        state: Optional[PokerTableState] = None,
    ) -> np.ndarray:
        """
        Draw visualization on image.
        
        Args:
            image: Input image
            state: Optional state to visualize (uses current if None)
            
        Returns:
            Image with visualizations
        """
        if state is None:
            state = self.current_state
        
        vis = image.copy()
        
        # Draw tracked objects
        for track in self.tracker.get_stable_tracks():
            x1, y1, x2, y2 = track.bbox
            
            # Color based on class
            if track.class_name == 'card':
                color = (0, 255, 0)  # Green
            elif track.class_name == 'chip':
                color = (0, 0, 255)  # Red
            elif track.class_name == 'player':
                color = (255, 0, 0)  # Blue
            else:
                color = (255, 255, 0)  # Cyan
            
            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{track.class_name} #{track.track_id}"
            if hasattr(track, 'classification') and track.classification:
                rank = getattr(track.classification, 'rank', '')
                suit = getattr(track.classification, 'suit', '')
                if rank and suit:
                    label += f": {rank}{suit[0]}"
            
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw table layout regions (if calibrated)
        if self.calibrator.current_layout:
            layout = self.calibrator.current_layout
            h, w = vis.shape[:2]
            
            # Draw table boundary
            tx1, ty1, tx2, ty2 = layout.table_bbox.to_pixel_coords(w, h)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
            cv2.putText(vis, "TABLE", (tx1, ty1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Draw board region
            bx1, by1, bx2, by2 = layout.board_region.to_pixel_coords(w, h)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            
            # Draw pot region
            px1, py1, px2, py2 = layout.pot_region.to_pixel_coords(w, h)
            cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 255, 0), 2)
            
            # Draw seat regions
            for seat in layout.seats:
                sx1, sy1, sx2, sy2 = seat.region.to_pixel_coords(w, h)
                color_seat = (128, 128, 128) if not seat.is_active else (255, 128, 0)
                cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color_seat, 1)
                cv2.putText(vis, seat.position.name, (sx1, sy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_seat, 1)
        
        # Draw state info
        info_y = 30
        cv2.putText(vis, f"Street: {state.street.value}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(vis, f"Cards: {len(state.board_cards)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(vis, f"FPS: {self.metrics.current_fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw calibration info
        if self.calibrator.current_layout:
            calib_stats = self.calibrator.get_stats()
            info_y += 25
            cv2.putText(vis, f"Layout v{calib_stats['layout_version']}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            info_y += 20
            cv2.putText(vis, f"Calib conf: {calib_stats['current_confidence']:.2f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw board cards
        if state.board_cards:
            cards_text = "Board: " + " ".join([c.short_name for c in state.board_cards])
            cv2.putText(vis, cards_text, (10, vis.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis
    
    def process_video(
        self,
        source: int | str,
        display: bool = True,
        max_frames: Optional[int] = None,
    ) -> List[PokerTableState]:
        """
        Process video from camera or file.
        
        Args:
            source: Camera index or video file path
            display: Show visualization window
            max_frames: Maximum frames to process (None = unlimited)
            
        Returns:
            List of processed states
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        results = []
        frame_count = 0
        
        self.start()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                state = self.process_frame(frame)
                results.append(state)
                frame_count += 1
                
                if display:
                    vis = self.visualize(frame, state)
                    cv2.imshow("Poker Vision", vis)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if max_frames and frame_count >= max_frames:
                    break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.stop()
        
        logger.info(f"Processed {frame_count} frames")
        return results


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('poker_vision.log'),
        ]
    )


# Example usage
if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    
    # Create pipeline with mock mode for testing
    config = PipelineConfig(
        use_mock=True,
        debug_mode=True,
        visualize=True,
    )
    
    pipeline = PokerVisionPipeline(config=config)
    
    # Test with sample image
    test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    state = pipeline.process_frame(test_image)
    print(f"Initial state: {state.to_dict()}")
    
    # Get metrics
    metrics = pipeline.get_metrics()
    print(f"Metrics: {metrics}")
