"""
vision/guardrails.py - Anti-Error Guardrails System

Comprehensive error prevention and recovery system:
- Confidence thresholding at multiple levels
- Temporal validation (changes must persist)
- Rollback mechanisms for invalid states
- Anomaly detection for impossible transitions
- Rate limiting to prevent flickering

This is the safety layer that ensures production reliability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Result of validation check."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """
    Result of a guardrail validation check.
    
    Attributes:
        status: Overall validation status
        checks_passed: Number of checks passed
        checks_failed: Number of checks failed
        warnings: List of warning messages
        errors: List of error messages
        should_rollback: Whether to rollback to previous state
        confidence_adjustment: Adjustment to apply to overall confidence
    """
    status: ValidationStatus = ValidationStatus.VALID
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    should_rollback: bool = False
    confidence_adjustment: float = 0.0
    
    def add_warning(self, message: str) -> None:
        """Add a warning."""
        self.warnings.append(message)
        self.checks_failed += 1
        if self.status == ValidationStatus.VALID:
            self.status = ValidationStatus.WARNING
    
    def add_error(self, message: str, critical: bool = False) -> None:
        """Add an error."""
        self.errors.append(message)
        self.checks_failed += 1
        if critical:
            self.status = ValidationStatus.CRITICAL
            self.should_rollback = True
        elif self.status != ValidationStatus.CRITICAL:
            self.status = ValidationStatus.INVALID
    
    def add_pass(self) -> None:
        """Record a passed check."""
        self.checks_passed += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "errors": self.errors,
            "should_rollback": self.should_rollback,
            "confidence_adjustment": self.confidence_adjustment,
        }


class Guardrails:
    """
    Anti-error guardrails system for poker vision pipeline.
    
    Features:
    - Multi-level confidence thresholding
    - Temporal consistency validation
    - State transition validation
    - Anomaly detection
    - Automatic rollback triggers
    - Flicker prevention
    """
    
    # Default thresholds
    DEFAULT_CONFIG = {
        # Confidence thresholds
        "min_detection_confidence": 0.3,
        "min_classification_confidence": 0.4,
        "min_fused_confidence": 0.5,
        "min_state_confidence": 0.6,
        
        # Temporal thresholds
        "min_frames_to_confirm": 3,
        "max_frames_without_detection": 10,
        "change_persistence_frames": 5,
        
        # Rate limiting
        "max_state_changes_per_second": 2,
        "min_time_between_card_changes": 1.0,
        
        # Anomaly detection
        "max_cards_on_board": 5,
        "max_duplicate_card_tolerance": 0,
        "max_position_jump_pixels": 100,
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize guardrails system.
        
        Args:
            config: Custom configuration overrides
            enable_logging: Enable detailed logging
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        self.enable_logging = enable_logging
        
        # State history for temporal validation
        self.state_history: deque = deque(maxlen=30)
        self.card_history: Dict[str, deque] = {}  # Track individual cards
        self.change_timestamps: deque = deque(maxlen=100)
        
        # Statistics
        self.validations_run = 0
        self.validations_passed = 0
        self.rollbacks_triggered = 0
        self.anomalies_detected = 0
        
        logger.info("Guardrails system initialized")
    
    def validate_frame(
        self,
        fused_detections: List[Any],
        current_state: Any,
        previous_state: Optional[Any] = None,
        frame_number: int = 0,
    ) -> ValidationResult:
        """
        Run all guardrail validations on a frame.
        
        Args:
            fused_detections: List of FusedDetection from fusion engine
            current_state: Current PokerTableState
            previous_state: Previous PokerTableState (for comparison)
            frame_number: Current frame number
            
        Returns:
            ValidationResult with all checks
        """
        self.validations_run += 1
        result = ValidationResult()
        
        # Run all validation checks
        self._validate_detection_confidence(fused_detections, result)
        self._validate_classification_confidence(fused_detections, result)
        self._validate_temporal_consistency(current_state, previous_state, result)
        self._validate_card_uniqueness(fused_detections, result)
        self._validate_state_transitions(current_state, previous_state, result)
        self._validate_rate_limits(result)
        self._validate_anomaly_detection(fused_detections, current_state, result)
        
        # Compute confidence adjustment
        if result.checks_failed > 0:
            result.confidence_adjustment = -0.1 * result.checks_failed
        
        # Record in history
        self._record_to_history(current_state, frame_number)
        
        if result.status == ValidationStatus.VALID:
            self.validations_passed += 1
        
        if result.should_rollback:
            self.rollbacks_triggered += 1
        
        if self.enable_logging and result.status != ValidationStatus.VALID:
            self._log_validation_result(result, frame_number)
        
        return result
    
    def _validate_detection_confidence(
        self,
        detections: List[Any],
        result: ValidationResult,
    ) -> None:
        """Validate detection confidence levels."""
        min_conf = self.config["min_detection_confidence"]
        
        low_conf_count = 0
        for det in detections:
            conf = getattr(det, 'detection_confidence', 0.5)
            if conf < min_conf:
                low_conf_count += 1
        
        if low_conf_count == 0:
            result.add_pass()
        elif low_conf_count <= len(detections) * 0.3:
            result.add_warning(f"{low_conf_count} detections below confidence threshold")
        else:
            result.add_error(
                f"Too many low-confidence detections: {low_conf_count}/{len(detections)}",
                critical=False
            )
    
    def _validate_classification_confidence(
        self,
        detections: List[Any],
        result: ValidationResult,
    ) -> None:
        """Validate classification confidence for cards."""
        min_conf = self.config["min_classification_confidence"]
        
        card_dets = [d for d in detections if getattr(d, 'class_name', '') == 'card']
        
        low_conf_cards = []
        for det in card_dets:
            conf = getattr(det, 'classification_confidence', 0.5)
            if conf < min_conf and getattr(det, 'rank', ''):
                low_conf_cards.append(f"{det.rank}?")
        
        if not low_conf_cards:
            result.add_pass()
        elif len(low_conf_cards) <= 1:
            result.add_warning(f"Low confidence classifications: {low_conf_cards}")
        else:
            result.add_error(f"Multiple uncertain card classifications: {low_conf_cards}")
    
    def _validate_temporal_consistency(
        self,
        current_state: Any,
        previous_state: Optional[Any],
        result: ValidationResult,
    ) -> None:
        """Validate temporal consistency between frames."""
        if previous_state is None:
            result.add_pass()
            return
        
        # Check for sudden large changes
        # Use card short_name for hashing (string representation)
        current_cards = set(
            c.short_name if hasattr(c, 'short_name') else str(c)
            for c in getattr(current_state, 'board_cards', [])
        )
        previous_cards = set(
            c.short_name if hasattr(c, 'short_name') else str(c)
            for c in getattr(previous_state, 'board_cards', [])
        )
        
        # Cards should not disappear suddenly
        removed_cards = previous_cards - current_cards
        if len(removed_cards) > 1:
            result.add_warning(f"Multiple cards disappeared: {removed_cards}")
        
        # Cards should not appear without street change
        added_cards = current_cards - previous_cards
        if len(added_cards) > 0:
            current_street = getattr(current_state, 'street', None)
            previous_street = getattr(previous_state, 'street', None)
            
            if current_street == previous_street and len(added_cards) > 1:
                result.add_error(
                    f"Cards appeared without street change: {added_cards}",
                    critical=True
                )
                self.anomalies_detected += 1
        else:
            result.add_pass()
    
    def _validate_card_uniqueness(
        self,
        detections: List[Any],
        result: ValidationResult,
    ) -> None:
        """Validate no duplicate cards are detected."""
        cards_seen = {}
        
        for det in detections:
            if getattr(det, 'class_name', '') != 'card':
                continue
            
            rank = getattr(det, 'rank', '')
            suit = getattr(det, 'suit', '')
            
            if not rank or not suit:
                continue
            
            card_key = f"{rank}{suit}"
            if card_key in cards_seen:
                result.add_error(
                    f"Duplicate card detected: {card_key}",
                    critical=True
                )
                self.anomalies_detected += 1
            else:
                cards_seen[card_key] = det
        
        if not cards_seen:
            result.add_pass()
    
    def _validate_state_transitions(
        self,
        current_state: Any,
        previous_state: Optional[Any],
        result: ValidationResult,
    ) -> None:
        """Validate poker rule compliance in state transitions."""
        if previous_state is None:
            result.add_pass()
            return
        
        # Valid street transitions
        valid_transitions = {
            "unknown": ["preflop", "flop"],
            "preflop": ["flop", "unknown"],
            "flop": ["turn", "unknown"],
            "turn": ["river", "unknown"],
            "river": ["showdown", "unknown"],
            "showdown": ["preflop", "unknown"],
        }
        
        current_street = getattr(current_state, 'street', 'unknown')
        previous_street = getattr(previous_state, 'street', 'unknown')
        
        if hasattr(current_street, 'value'):
            current_street = current_street.value
        if hasattr(previous_street, 'value'):
            previous_street = previous_street.value
        
        if current_street != previous_street:
            allowed = valid_transitions.get(previous_street, [])
            if current_street not in allowed:
                result.add_error(
                    f"Invalid street transition: {previous_street} -> {current_street}",
                    critical=True
                )
                self.anomalies_detected += 1
            else:
                result.add_pass()
        else:
            result.add_pass()
    
    def _validate_rate_limits(self, result: ValidationResult) -> None:
        """Validate rate limits to prevent flickering."""
        now = time.time()
        max_changes = self.config["max_state_changes_per_second"]
        
        # Clean old timestamps
        while self.change_timestamps and now - self.change_timestamps[0] > 1.0:
            self.change_timestamps.popleft()
        
        if len(self.change_timestamps) <= max_changes:
            result.add_pass()
        else:
            result.add_warning("Rate limit exceeded - possible flickering")
    
    def _validate_anomaly_detection(
        self,
        detections: List[Any],
        current_state: Any,
        result: ValidationResult,
    ) -> None:
        """Detect anomalous patterns."""
        # Check for too many cards
        card_dets = [d for d in detections if getattr(d, 'class_name', '') == 'card']
        max_cards = self.config["max_cards_on_board"]
        
        if len(card_dets) > max_cards + 2:  # Allow some tolerance for hole cards
            result.add_warning(f"Unusual number of cards detected: {len(card_dets)}")
        
        # Check for impossible card combinations
        ranks = [getattr(d, 'rank', '') for d in card_dets if getattr(d, 'rank', '')]
        
        # More than 4 of same rank is impossible (unless bug)
        from collections import Counter
        rank_counts = Counter(ranks)
        for rank, count in rank_counts.items():
            if count > 4:
                result.add_error(
                    f"Impossible: {count} cards of rank {rank}",
                    critical=True
                )
                self.anomalies_detected += 1
                break
        else:
            result.add_pass()
    
    def _record_to_history(self, state: Any, frame_number: int) -> None:
        """Record state to history for temporal analysis."""
        self.state_history.append({
            "frame": frame_number,
            "timestamp": time.time(),
            "cards": getattr(state, 'board_cards', []).copy() if hasattr(state, 'board_cards') else [],
            "street": getattr(state, 'street', None),
        })
    
    def _log_validation_result(
        self,
        result: ValidationResult,
        frame_number: int,
    ) -> None:
        """Log validation result."""
        log_msg = f"[Frame {frame_number}] Validation: {result.status.value}"
        
        if result.warnings:
            log_msg += f" | Warnings: {result.warnings}"
        if result.errors:
            log_msg += f" | Errors: {result.errors}"
        
        if result.status == ValidationStatus.CRITICAL:
            logger.error(log_msg)
        elif result.status == ValidationStatus.INVALID:
            logger.warning(log_msg)
        else:
            logger.debug(log_msg)
    
    def should_accept_change(
        self,
        change_type: str,
        old_value: Any,
        new_value: Any,
        frames_confirmed: int = 0,
    ) -> bool:
        """
        Determine if a detected change should be accepted.
        
        Args:
            change_type: Type of change ("card", "street", "stack", etc.)
            old_value: Previous value
            new_value: New value
            frames_confirmed: Number of frames this change has been seen
            
        Returns:
            True if change should be accepted
        """
        if old_value == new_value:
            return True
        
        min_frames = self.config["min_frames_to_confirm"]
        
        if frames_confirmed >= min_frames:
            # Change has persisted long enough
            self.change_timestamps.append(time.time())
            return True
        
        return False
    
    def get_stable_value(
        self,
        value_history: List[Tuple[Any, int]],
        default: Any = None,
    ) -> Any:
        """
        Get the most stable value from history.
        
        Args:
            value_history: List of (value, frame_count) tuples
            default: Default value if history is empty
            
        Returns:
            Most stable (persistent) value
        """
        if not value_history:
            return default
        
        # Weight by persistence
        weighted_values = {}
        for value, frames in value_history:
            if value not in weighted_values:
                weighted_values[value] = 0
            weighted_values[value] += frames
        
        if not weighted_values:
            return default
        
        return max(weighted_values.keys(), key=lambda v: weighted_values[v])
    
    def record_change(self) -> None:
        """Record a state change for rate limiting."""
        self.change_timestamps.append(time.time())
    
    def reset(self) -> None:
        """Reset guardrails state."""
        self.state_history.clear()
        self.card_history.clear()
        self.change_timestamps.clear()
        logger.info("Guardrails reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        return {
            "validations_run": self.validations_run,
            "validations_passed": self.validations_passed,
            "rollbacks_triggered": self.rollbacks_triggered,
            "anomalies_detected": self.anomalies_detected,
            "pass_rate": self.validations_passed / max(1, self.validations_run),
            "config": self.config,
        }
