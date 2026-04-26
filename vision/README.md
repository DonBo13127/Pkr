# Production-Grade Poker Vision System

A high-reliability, real-time poker table vision system with industrial robustness.

## Architecture Overview

```
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
```

## Features

### 1. YOLO Detection Module (`yolo_detector.py`)
- GPU-optimized object detection
- Detects: cards, chips, players, dealer button, pot, UI elements
- Per-class confidence thresholds
- Batch processing support
- Mock mode for testing

### 2. CNN Card Classification (`card_classifier.py`)
- **NOT OCR-based** - Uses trained CNN for visual pattern recognition
- Multi-task network (rank + suit classification)
- Handles partial visibility and occlusion
- Robust to rotation, scale, and lighting variations
- Fallback heuristics when model unavailable

### 3. Temporal Tracking (`tracker.py`)
- **CRITICAL for stability** - prevents flickering
- IoU-based data association
- Kalman-like motion prediction
- Track persistence across frames
- Confidence decay for missed detections
- Exponential smoothing for bounding boxes

### 4. Fusion Engine (`fusion.py`)
- **No single model decides alone**
- Weighted fusion of:
  - YOLO detection confidence (40%)
  - CNN classification confidence (35%)
  - Temporal consistency (25%)
- Adaptive weighting based on conditions
- Outlier rejection

### 5. State Machine (`state_machine.py`)
- Maintains complete poker table state
- Enforces poker rules:
  - Valid street transitions (preflop → flop → turn → river)
  - Card count limits per street
  - No duplicate cards
  - Stack consistency
- Rollback capability for invalid states

### 6. Guardrails (`guardrails.py`)
- **Anti-error system**
- Multi-level confidence thresholding
- Temporal validation (changes must persist)
- Anomaly detection
- Rate limiting to prevent flickering
- Automatic rollback triggers

### 7. Main Pipeline (`pipeline.py`)
- Async processing with thread pool
- Real-time performance (>= 20 FPS target)
- Frame skipping under load
- Comprehensive metrics
- Visualization overlay

## Installation

```bash
# Core dependencies
pip install opencv-python-headless numpy

# For GPU acceleration (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For YOLO detection (optional)
pip install ultralytics

# Full installation
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from vision import PokerVisionPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    use_mock=True,          # Use mock models for testing
    debug_mode=True,        # Enable debug logging
    max_fps=30.0,           # Target FPS
)

# Create pipeline
pipeline = PokerVisionPipeline(config=config)

# Process frame
import cv2
image = cv2.imread("poker_table.png")
state = pipeline.process_frame(image)

# Access state
print(f"Street: {state.street.value}")
print(f"Board cards: {state.get_board_cards()}")
print(f"Confidence: {state.confidence:.2f}")

# Get metrics
metrics = pipeline.get_metrics()
print(f"FPS: {metrics['pipeline']['current_fps']:.1f}")
```

### Video Processing

```python
# Process video file or camera
results = pipeline.process_video(
    source="poker_video.mp4",  # or 0 for webcam
    display=True,              # Show visualization
    max_frames=1000,           # Optional limit
)
```

### Advanced Configuration

```python
config = PipelineConfig(
    # Model paths
    yolo_model_path="models/poker_yolov8.pt",
    classifier_model_path="models/card_classifier.pth",
    
    # Device
    device="cuda",  # or "cpu" or "auto"
    
    # Performance
    max_fps=30.0,
    enable_async=True,
    
    # Thresholds
    min_detection_confidence=0.4,
    min_classification_confidence=0.4,
    min_fused_confidence=0.5,
    
    # Tracking
    max_tracks=100,
    track_confirm_frames=3,
    
    # Guardrails
    enable_rollback=True,
    enable_guardrails=True,
    
    # Debug
    debug_mode=False,
    visualize=False,
)
```

## Running Tests

```bash
# Run all tests
python -m vision.test_pipeline

# Run specific test
python -c "from vision.test_pipeline import test_tracker; test_tracker()"
```

## Output Format

### PokerTableState

```python
{
    "street": "flop",
    "pot": 150.0,
    "players": {
        "0": {
            "player_id": 0,
            "position": "BTN",
            "stack": 1000.0,
            "hole_cards": ["Ah", "Ks"],
            "is_active": True
        }
    },
    "board_cards": [
        {"rank": "Q", "suit": "diamonds", "short_name": "Qd"},
        {"rank": "J", "suit": "spades", "short_name": "Js"},
        {"rank": "T", "suit": "hearts", "short_name": "Th"}
    ],
    "confidence": 0.85,
    "is_valid": True,
    "validation_errors": []
}
```

## Performance Metrics

The pipeline provides comprehensive metrics:

```python
metrics = pipeline.get_metrics()
print(metrics)

# Output:
{
    "pipeline": {
        "frames_processed": 1000,
        "current_fps": 28.5,
        "avg_processing_time_ms": 35.2,
        "detection_time_ms": 15.3,
        "classification_time_ms": 12.1,
        "tracking_time_ms": 2.5,
        "fusion_time_ms": 1.2,
        "state_update_time_ms": 4.1
    },
    "tracker": {
        "total_tracks": 15,
        "stable_tracks": 12,
        "frame_count": 1000
    },
    "guardrails": {
        "validations_run": 1000,
        "pass_rate": 0.97,
        "rollbacks_triggered": 3
    }
}
```

## Model Training

### YOLO Detection Model

Train a custom YOLOv8 model for poker table detection:

```bash
# Prepare dataset in YOLO format
# Train
yolo detect train data=poker_dataset.yaml model=yolov8n.pt epochs=100
```

### Card Classification Model

Train the card classifier:

```python
# See scripts/train_classifier.py for training code
python scripts/train_classifier.py --data_dir datasets/cards
```

## Troubleshooting

### Low FPS
- Enable `use_mock=True` for testing without models
- Reduce `max_fps` target
- Use GPU (`device="cuda"`)
- Enable async processing

### Flickering Detections
- Increase `track_confirm_frames`
- Increase `min_fused_confidence`
- Check guardrails `change_persistence_frames`

### False Card Classifications
- Retrain classifier with more diverse data
- Increase `min_classification_confidence`
- Ensure good lighting and image quality

## License

Proprietary - All rights reserved.
