# Poker Vision System - Test Suite

This directory contains comprehensive tests for the production-grade poker vision pipeline.

## Test Files

### `test_pipeline.py` (Main Test Suite)
Run with: `python -m vision.test_pipeline`

Tests all core components:
- Table Calibration
- YOLO Detector  
- Card Classifier
- Tracker
- Fusion Engine
- State Machine
- Guardrails
- Full Pipeline Integration
- Sample Images Processing

### `test_stability.py` (Temporal Stability Tests)
Run with: `python tests/test_stability.py --input tests/screenshots --frames 15`

Tests for flicker prevention and temporal consistency:
- Processes same image multiple times
- Measures stability score (0-1)
- Reports FPS performance
- Generates detailed stability report

### `test_cards.py` (Card Detection Tests)
Run with: `python tests/test_cards.py --input tests/screenshots`

Tests card detection accuracy:
- Detection rate on real screenshots
- Card count distribution
- Per-image results with detected cards

## Screenshot Testing

Place your poker table screenshots in `tests/screenshots/` directory.

Supported formats: `.png`, `.jpg`

### Running All Tests

```bash
# Run main test suite
python -m vision.test_pipeline

# Run stability tests on all screenshots
python tests/test_stability.py --input tests/screenshots --frames 15

# Run card detection tests
python tests/test_cards.py --input tests/screenshots

# Save reports to files
python tests/test_stability.py -i tests/screenshots -o logs/stability_report.txt
python tests/test_cards.py -i tests/screenshots -o logs/card_report.txt
```

## Output Format

Each test produces structured output:

```json
{
  "cards": ["As", "Kh", "Qd"],
  "players": [...],
  "pot": value,
  "confidence": 0.85,
  "stable": true
}
```

## Metrics Tracked

- **Detection Accuracy**: Cards, chips, players detected
- **Classification Accuracy**: Rank and suit prediction
- **Stability Score**: Temporal consistency (no flickering)
- **Confidence Score**: Combined model confidence
- **FPS Performance**: Real-time capability
- **Validation Pass Rate**: Guardrail effectiveness

## Expected Results

With mock mode (default for testing):
- Stability Score: > 0.7 (GOOD), > 0.9 (EXCELLENT)
- Validity Ratio: ~100%
- FPS: Varies by hardware (mock inference is slower)

With real models:
- Expect significantly higher FPS (>20 target)
- Better card classification accuracy
- More reliable detections
