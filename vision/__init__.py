"""
vision/__init__.py - Production-Grade Poker Vision System

A high-reliability, real-time poker table vision system with industrial robustness.
Modular architecture with YOLOv8 detection, CNN classification, temporal tracking,
fusion engine, state machine, and comprehensive guardrails.

Author: Senior CV Engineer
Version: 1.0.0
"""

from .yolo_detector import YOLODetector, DetectionResult
from .card_classifier import CardClassifier, CardClassificationResult
from .tracker import Tracker, TrackedObject
from .fusion import FusionEngine, FusedDetection
from .state_machine import StateMachine, PokerTableState
from .guardrails import Guardrails, ValidationResult
from .pipeline import PokerVisionPipeline, PipelineConfig, PipelineMetrics

__version__ = "1.0.0"
__all__ = [
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
]
