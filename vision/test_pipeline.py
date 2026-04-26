"""
vision/test_pipeline.py - Test Suite for Poker Vision Pipeline

Comprehensive tests for all pipeline components.
Run with: python -m vision.test_pipeline
"""

from __future__ import annotations

import logging
import time
import sys
from pathlib import Path

import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_yolo_detector():
    """Test YOLO detector module."""
    print("\n" + "="*60)
    print("Testing YOLO Detector")
    print("="*60)
    
    from .yolo_detector import YOLODetector, DetectionResult
    
    # Create mock detector
    detector = YOLODetector(use_mock=True)
    
    # Create test image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(test_image)
    
    print(f"  Detections found: {len(detections)}")
    for det in detections[:5]:  # Show first 5
        print(f"    - {det.class_name}: bbox={det.bbox}, conf={det.confidence:.2f}")
    
    # Test batch detection
    batch_results = detector.detect_batch([test_image, test_image])
    print(f"  Batch detection: {[len(r) for r in batch_results]} detections")
    
    # Get model info
    info = detector.get_model_info()
    print(f"  Model info: device={info['device']}, use_mock={info['use_mock']}")
    
    assert isinstance(detections, list), "Detections should be a list"
    print("  ✓ YOLO Detector tests passed")
    
    return True


def test_card_classifier():
    """Test card classifier module."""
    print("\n" + "="*60)
    print("Testing Card Classifier")
    print("="*60)
    
    from .card_classifier import CardClassifier, VALID_RANKS, VALID_SUITS
    
    # Create mock classifier
    classifier = CardClassifier(use_mock=True)
    
    # Create test card image (white rectangle)
    card_image = np.ones((100, 70, 3), dtype=np.uint8) * 200
    
    # Classify
    result = classifier.classify(card_image)
    
    print(f"  Classification result:")
    print(f"    Rank: {result.rank}")
    print(f"    Suit: {result.suit}")
    print(f"    Rank confidence: {result.rank_confidence:.2f}")
    print(f"    Suit confidence: {result.suit_confidence:.2f}")
    print(f"    Is valid: {result.is_valid}")
    
    # Test batch classification
    batch_results = classifier.classify_batch([card_image, card_image])
    print(f"  Batch classification: {len(batch_results)} results")
    
    # Get model info
    info = classifier.get_model_info()
    print(f"  Model info: device={info['device']}, torch_available={info['torch_available']}")
    
    assert result.rank in VALID_RANKS or result.rank == "", "Rank should be valid or empty"
    assert result.suit in VALID_SUITS or result.suit == "", "Suit should be valid or empty"
    print("  ✓ Card Classifier tests passed")
    
    return True


def test_tracker():
    """Test tracker module."""
    print("\n" + "="*60)
    print("Testing Tracker")
    print("="*60)
    
    from .tracker import Tracker, TrackedObject
    from .yolo_detector import DetectionResult
    
    tracker = Tracker(max_tracks=50)
    
    # Create mock detections
    detections = [
        DetectionResult(
            class_id=0,
            class_name="card",
            bbox=(100, 100, 150, 200),
            confidence=0.8,
        ),
        DetectionResult(
            class_id=0,
            class_name="card",
            bbox=(200, 100, 250, 200),
            confidence=0.75,
        ),
    ]
    
    # Update tracker
    tracks = tracker.update(detections, timestamp=time.time())
    
    print(f"  Tracks after first update: {len(tracks)}")
    for track in tracks:
        print(f"    - Track #{track.track_id}: {track.class_name}, frames_seen={track.frames_seen}")
    
    # Simulate second frame with same objects
    time.sleep(0.01)
    detections2 = [
        DetectionResult(
            class_id=0,
            class_name="card",
            bbox=(102, 102, 152, 202),  # Slightly moved
            confidence=0.82,
        ),
        DetectionResult(
            class_id=0,
            class_name="card",
            bbox=(202, 102, 252, 202),
            confidence=0.78,
        ),
    ]
    
    tracks2 = tracker.update(detections2, timestamp=time.time())
    
    print(f"  Tracks after second update: {len(tracks2)}")
    for track in tracks2:
        print(f"    - Track #{track.track_id}: stable={track.is_stable}, temporal_conf={track.temporal_confidence:.2f}")
    
    # Get stats
    stats = tracker.get_stats()
    print(f"  Tracker stats: total_tracks={stats['total_tracks']}, frame_count={stats['frame_count']}")
    
    assert len(tracks2) >= 1, "Should have at least one track"
    print("  ✓ Tracker tests passed")
    
    return True


def test_fusion_engine():
    """Test fusion engine module."""
    print("\n" + "="*60)
    print("Testing Fusion Engine")
    print("="*60)
    
    from .fusion import FusionEngine, FusedDetection
    from .tracker import TrackedObject
    from collections import deque
    
    fusion = FusionEngine()
    
    # Create mock tracked objects
    track1 = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="card",
        bbox=(100, 100, 150, 200),
        confidence=0.8,
    )
    track1.frames_seen = 5
    track1.frames_missed = 0
    
    track2 = TrackedObject(
        track_id=2,
        class_id=1,
        class_name="chip",
        bbox=(300, 300, 340, 340),
        confidence=0.7,
    )
    track2.frames_seen = 3
    track2.frames_missed = 0
    
    tracks = [track1, track2]
    
    # Fuse
    fused = fusion.fuse(tracks)
    
    print(f"  Fused detections: {len(fused)}")
    for det in fused:
        print(f"    - {det.class_name} #{det.track_id}:")
        print(f"        Detection conf: {det.detection_confidence:.2f}")
        print(f"        Temporal conf: {det.temporal_confidence:.2f}")
        print(f"        Fused conf: {det.fused_confidence:.2f}")
        print(f"        Is reliable: {det.is_reliable}")
    
    # Filter unreliable
    reliable = fusion.filter_unreliable(fused)
    print(f"  Reliable detections after filtering: {len(reliable)}")
    
    # Get top detections by class
    top_cards = fusion.get_top_detections(fused, class_name="card", top_k=1)
    print(f"  Top card detection: {len(top_cards)} results")
    
    assert len(fused) == 2, "Should have 2 fused detections"
    print("  ✓ Fusion Engine tests passed")
    
    return True


def test_state_machine():
    """Test state machine module."""
    print("\n" + "="*60)
    print("Testing State Machine")
    print("="*60)
    
    from .state_machine import StateMachine, PokerTableState, Street
    from .fusion import FusedDetection
    
    state_machine = StateMachine(max_players=9, enable_rollback=True)
    
    # Create mock fused detections
    fused_dets = []
    
    # Initial state (preflop)
    state1 = state_machine.update(fused_dets, frame_number=1)
    print(f"  Initial state: street={state1.street.value}, cards={len(state1.board_cards)}")
    
    # Simulate flop (3 cards detected)
    fused_dets_with_cards = [
        FusedDetection(
            track_id=i,
            class_id=0,
            class_name="card",
            bbox=(100+i*50, 100, 150+i*50, 200),
            rank=rank,
            suit=suit,
            fused_confidence=0.8,
            is_reliable=True,
        )
        for i, (rank, suit) in enumerate([
            ("A", "spades"),
            ("K", "hearts"),
            ("Q", "diamonds"),
        ])
    ]
    
    state2 = state_machine.update(fused_dets_with_cards, frame_number=2)
    print(f"  After flop: street={state2.street.value}, cards={len(state2.board_cards)}")
    print(f"    Board: {[c.short_name for c in state2.board_cards]}")
    
    # Get stats
    stats = state_machine.get_stats()
    print(f"  State machine stats: transitions={stats['transition_count']}, rollbacks={stats['rollback_count']}")
    
    assert state2.street in [Street.FLOP, Street.UNKNOWN], "Street should be flop or unknown"
    print("  ✓ State Machine tests passed")
    
    return True


def test_guardrails():
    """Test guardrails module."""
    print("\n" + "="*60)
    print("Testing Guardrails")
    print("="*60)
    
    from .guardrails import Guardrails, ValidationResult, ValidationStatus
    from .fusion import FusedDetection
    from .state_machine import PokerTableState, Street
    
    guardrails = Guardrails(enable_logging=True)
    
    # Create mock state
    current_state = PokerTableState(street=Street.FLOP)
    previous_state = PokerTableState(street=Street.PREFLOP)
    
    # Create mock detections
    detections = [
        FusedDetection(
            track_id=1,
            class_id=0,
            class_name="card",
            bbox=(100, 100, 150, 200),
            rank="A",
            suit="spades",
            detection_confidence=0.8,
            classification_confidence=0.75,
            temporal_confidence=0.85,
            fused_confidence=0.8,
            is_reliable=True,
        ),
    ]
    
    # Validate
    result = guardrails.validate_frame(
        fused_detections=detections,
        current_state=current_state,
        previous_state=previous_state,
        frame_number=1,
    )
    
    print(f"  Validation result: status={result.status.value}")
    print(f"    Checks passed: {result.checks_passed}")
    print(f"    Checks failed: {result.checks_failed}")
    print(f"    Warnings: {len(result.warnings)}")
    print(f"    Errors: {len(result.errors)}")
    print(f"    Should rollback: {result.should_rollback}")
    
    # Test change acceptance
    should_accept = guardrails.should_accept_change(
        change_type="card",
        old_value="Ah",
        new_value="Ah",
        frames_confirmed=5,
    )
    print(f"  Should accept unchanged card: {should_accept}")
    
    # Get stats
    stats = guardrails.get_stats()
    print(f"  Guardrails stats: validations={stats['validations_run']}, pass_rate={stats['pass_rate']:.2f}")
    
    assert result.status in [ValidationStatus.VALID, ValidationStatus.WARNING], "Should not be critical"
    print("  ✓ Guardrails tests passed")
    
    return True


def test_full_pipeline():
    """Test complete pipeline integration."""
    print("\n" + "="*60)
    print("Testing Full Pipeline Integration")
    print("="*60)
    
    from .pipeline import PokerVisionPipeline, PipelineConfig
    
    # Create pipeline with mock mode
    config = PipelineConfig(
        use_mock=True,
        debug_mode=True,
        max_fps=30.0,
    )
    
    pipeline = PokerVisionPipeline(config=config)
    
    # Create test image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Process single frame
    start_time = time.time()
    state = pipeline.process_frame(test_image)
    process_time = time.time() - start_time
    
    print(f"  Frame processed in {process_time*1000:.1f}ms")
    print(f"  FPS: {1/process_time:.1f}")
    print(f"  State: street={state.street.value}, valid={state.is_valid}")
    
    # Process multiple frames to test temporal stability
    states = []
    for i in range(10):
        state = pipeline.process_frame(test_image)
        states.append(state)
    
    # Check metrics
    metrics = pipeline.get_metrics()
    print(f"  Pipeline metrics:")
    print(f"    Frames processed: {metrics['pipeline']['frames_processed']}")
    print(f"    Current FPS: {metrics['pipeline']['current_fps']:.1f}")
    print(f"    Avg processing time: {metrics['pipeline']['avg_processing_time_ms']:.1f}ms")
    
    # Test visualization
    vis_image = pipeline.visualize(test_image, state)
    print(f"  Visualization generated: shape={vis_image.shape}")
    
    # Test reset
    pipeline.reset()
    print("  Pipeline reset successfully")
    
    assert state is not None, "State should not be None"
    assert metrics['pipeline']['frames_processed'] > 0, "Should have processed frames"
    print("  ✓ Full Pipeline tests passed")
    
    return True


def test_with_sample_images():
    """Test pipeline with sample capture images if available."""
    print("\n" + "="*60)
    print("Testing with Sample Images")
    print("="*60)
    
    from .pipeline import PokerVisionPipeline, PipelineConfig
    
    # Find sample images
    sample_dir = Path(__file__).parent.parent
    sample_images = list(sample_dir.glob("capture_*.png"))
    
    if not sample_images:
        print("  No sample images found, skipping this test")
        return True
    
    print(f"  Found {len(sample_images)} sample images")
    
    # Create pipeline
    config = PipelineConfig(
        use_mock=True,
        debug_mode=False,
    )
    pipeline = PokerVisionPipeline(config=config)
    
    # Process each image
    for img_path in sample_images[:3]:  # Test first 3
        print(f"  Processing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    Failed to load image")
            continue
        
        state = pipeline.process_frame(image)
        
        print(f"    State: street={state.street.value}, confidence={state.confidence:.2f}")
        
        # Generate visualization
        vis = pipeline.visualize(image, state)
        
        # Save visualization
        vis_path = sample_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(vis_path), vis)
        print(f"    Saved visualization: {vis_path.name}")
    
    print("  ✓ Sample Image tests completed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("POKER VISION PIPELINE - TEST SUITE")
    print("="*70)
    
    tests = [
        ("YOLO Detector", test_yolo_detector),
        ("Card Classifier", test_card_classifier),
        ("Tracker", test_tracker),
        ("Fusion Engine", test_fusion_engine),
        ("State Machine", test_state_machine),
        ("Guardrails", test_guardrails),
        ("Full Pipeline", test_full_pipeline),
        ("Sample Images", test_with_sample_images),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASS" if success else "FAIL"
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results[name] = f"ERROR: {e}"
            failed += 1
            logger.exception(f"Test {name} failed with exception")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        status_icon = "✓" if result == "PASS" else "✗"
        print(f"  {status_icon} {name}: {result}")
    
    print(f"\n  Total: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
