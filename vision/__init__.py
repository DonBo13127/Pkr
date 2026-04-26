"""
vision/__init__.py - Production-Grade Poker Vision System

A high-reliability, real-time poker table vision system with industrial robustness.
Modular architecture with YOLOv8 detection, CNN classification, temporal tracking,
fusion engine, state machine, comprehensive guardrails, and dynamic table calibration.

Author: Senior CV Engineer
Version: 1.1.0
"""

from .yolo_detector import YOLODetector, DetectionResult
from .card_classifier import CardClassifier, CardClassificationResult
from .tracker import Tracker, TrackedObject
from .fusion import FusionEngine, FusedDetection
from .state_machine import StateMachine, PokerTableState
from .guardrails import Guardrails, ValidationResult
from .pipeline import PokerVisionPipeline, PipelineConfig, PipelineMetrics
from .table_calibration import (
    TableCalibrator,
    TableLayout,
    NormalizedBox,
    SeatInfo,
    SeatPosition,
)

__version__ = "1.1.0"
__all__ = [
    # Core components
    "YOLODetector",
    "DetectionResult",
    "CardClassifier",
    "CardClassificationResult",
    "Tracker",
    "TrackedObject",
    "FusionEngine",
    "FusedDetection",
    "StateMachine",
    "PokerTableState",
    "Guardrails",
    "ValidationResult",
    "PokerVisionPipeline",
    "PipelineConfig",
    "PipelineMetrics",
    # Table calibration
    "TableCalibrator",
    "TableLayout",
    "NormalizedBox",
    "SeatInfo",
    "SeatPosition",
]
