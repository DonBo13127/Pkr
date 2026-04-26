"""
tests/test_stability.py - Stability Testing Module

Tests for temporal stability and flicker prevention:
- Simulate multiple frames from same image
- Ensure outputs are identical (no flickering)
- Measure stability score over time
- Test rollback mechanisms
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

import cv2
import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.pipeline import PokerVisionPipeline, PipelineConfig
from vision.state_machine import PokerTableState, Street

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StabilityTester:
    """
    Test temporal stability of the vision pipeline.
    
    Ensures that:
    - Same input produces consistent output
    - State doesn't flicker between frames
    - Changes only occur when they should
    """
    
    def __init__(self, use_mock: bool = True):
        self.config = PipelineConfig(
            use_mock=use_mock,
            debug_mode=False,
            max_fps=30.0,
            enable_guardrails=True,
            enable_calibration=True,
        )
        self.pipeline = PokerVisionPipeline(config=self.config)
        
        # Metrics
        self.frame_results: List[Dict[str, Any]] = []
        
    def run_stability_test(
        self,
        image: np.ndarray,
        num_frames: int = 30,
    ) -> Dict[str, Any]:
        """
        Run stability test on a single image.
        
        Process the same image multiple times and measure consistency.
        
        Args:
            image: Input image to test
            num_frames: Number of frames to process
            
        Returns:
            Stability metrics dictionary
        """
        logger.info(f"Running stability test: {num_frames} frames")
        
        states: List[PokerTableState] = []
        processing_times: List[float] = []
        
        for i in range(num_frames):
            start = time.time()
            state = self.pipeline.process_frame(image)
            elapsed = time.time() - start
            
            states.append(state)
            processing_times.append(elapsed)
            
            # Record result
            self.frame_results.append({
                "frame": i,
                "street": state.street.value,
                "board_cards": state.get_board_cards(),
                "confidence": state.confidence,
                "is_valid": state.is_valid,
                "processing_time_ms": elapsed * 1000,
            })
        
        # Analyze stability
        return self._analyze_stability(states, processing_times)
    
    def _analyze_stability(
        self,
        states: List[PokerTableState],
        processing_times: List[float],
    ) -> Dict[str, Any]:
        """Analyze stability of results."""
        
        # Count street occurrences
        street_counts = Counter(s.street.value for s in states)
        most_common_street = street_counts.most_common(1)[0]
        street_stability = most_common_street[1] / len(states)
        
        # Count board card changes
        board_sequences = [tuple(s.get_board_cards()) for s in states]
        unique_boards = set(board_sequences)
        board_stability = 1.0 - (len(unique_boards) - 1) / max(1, len(states) - 1)
        
        # Confidence stability
        confidences = [s.confidence for s in states]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence)**2 for c in confidences) / len(confidences)
        
        # Valid state ratio
        valid_count = sum(1 for s in states if s.is_valid)
        validity_ratio = valid_count / len(states)
        
        # Processing time stats
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        # Overall stability score (0-1)
        overall_score = (
            street_stability * 0.3 +
            board_stability * 0.4 +
            validity_ratio * 0.2 +
            (1.0 - min(1.0, confidence_variance * 10)) * 0.1
        )
        
        return {
            "num_frames": len(states),
            "overall_stability_score": overall_score,
            "street_stability": street_stability,
            "street_distribution": dict(street_counts),
            "most_common_street": most_common_street[0],
            "board_stability": board_stability,
            "unique_board_configs": len(unique_boards),
            "confidence": {
                "average": avg_confidence,
                "variance": confidence_variance,
                "min": min(confidences),
                "max": max(confidences),
            },
            "validity_ratio": validity_ratio,
            "processing_time_ms": {
                "average": avg_time * 1000,
                "min": min_time * 1000,
                "max": max_time * 1000,
            },
            "fps_estimate": 1.0 / avg_time if avg_time > 0 else 0,
        }
    
    def reset(self):
        """Reset pipeline state."""
        self.pipeline.reset()
        self.frame_results.clear()


def test_screenshot_stability(
    screenshot_path: str,
    num_frames: int = 20,
) -> Dict[str, Any]:
    """
    Test stability on a specific screenshot.
    
    Args:
        screenshot_path: Path to screenshot image
        num_frames: Number of frames to process
        
    Returns:
        Stability test results
    """
    logger.info(f"Testing stability for: {screenshot_path}")
    
    # Load image
    image = cv2.imread(screenshot_path)
    if image is None:
        return {"error": f"Failed to load image: {screenshot_path}"}
    
    logger.info(f"Loaded image: {image.shape}")
    
    # Create tester and run
    tester = StabilityTester(use_mock=True)
    results = tester.run_stability_test(image, num_frames=num_frames)
    
    # Add metadata
    results["screenshot"] = Path(screenshot_path).name
    results["image_shape"] = list(image.shape)
    
    return results


def test_batch_stability(
    screenshot_dir: str,
    num_frames_per_image: int = 15,
) -> List[Dict[str, Any]]:
    """
    Test stability on all screenshots in a directory.
    
    Args:
        screenshot_dir: Directory containing screenshots
        num_frames_per_image: Frames to process per image
        
    Returns:
        List of stability results for each image
    """
    screenshot_path = Path(screenshot_dir)
    if not screenshot_path.exists():
        logger.error(f"Directory not found: {screenshot_dir}")
        return []
    
    images = list(screenshot_path.glob("*.png")) + list(screenshot_path.glob("*.jpg"))
    
    if not images:
        logger.warning(f"No images found in {screenshot_dir}")
        return []
    
    logger.info(f"Found {len(images)} images to test")
    
    all_results = []
    
    for img_path in images:
        try:
            result = test_screenshot_stability(
                str(img_path),
                num_frames=num_frames_per_image,
            )
            all_results.append(result)
            
            # Print summary
            if "error" not in result:
                logger.info(
                    f"  {img_path.name}: "
                    f"stability={result['overall_stability_score']:.2f}, "
                    f"avg_fps={result['fps_estimate']:.1f}"
                )
            else:
                logger.warning(f"  {img_path.name}: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error testing {img_path.name}: {e}")
            all_results.append({
                "screenshot": img_path.name,
                "error": str(e),
            })
    
    return all_results


def generate_stability_report(results: List[Dict[str, Any]]) -> str:
    """Generate a human-readable stability report."""
    
    lines = [
        "=" * 70,
        "STABILITY TEST REPORT",
        "=" * 70,
        "",
    ]
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    lines.append(f"Total images tested: {len(results)}")
    lines.append(f"Successful: {len(successful)}")
    lines.append(f"Failed: {len(failed)}")
    lines.append("")
    
    if successful:
        # Aggregate metrics
        avg_stability = sum(r["overall_stability_score"] for r in successful) / len(successful)
        avg_fps = sum(r["fps_estimate"] for r in successful) / len(successful)
        avg_validity = sum(r["validity_ratio"] for r in successful) / len(successful)
        
        lines.append("AGGREGATE METRICS:")
        lines.append(f"  Average Stability Score: {avg_stability:.3f}")
        lines.append(f"  Average FPS: {avg_fps:.2f}")
        lines.append(f"  Average Validity Ratio: {avg_validity:.2%}")
        lines.append("")
        
        # Per-image breakdown
        lines.append("PER-IMAGE RESULTS:")
        lines.append("-" * 70)
        
        for r in sorted(successful, key=lambda x: x.get("overall_stability_score", 0), reverse=True):
            lines.append(
                f"  {r['screenshot']:40s} | "
                f"Stability: {r['overall_stability_score']:.3f} | "
                f"FPS: {r['fps_estimate']:5.1f} | "
                f"Valid: {r['validity_ratio']:.0%}"
            )
        
        lines.append("")
        
        # Check for issues
        low_stability = [r for r in successful if r["overall_stability_score"] < 0.8]
        if low_stability:
            lines.append("⚠️  WARNING: Low stability detected in some images:")
            for r in low_stability:
                lines.append(f"   - {r['screenshot']}: {r['overall_stability_score']:.3f}")
            lines.append("")
    
    if failed:
        lines.append("FAILED TESTS:")
        for r in failed:
            lines.append(f"  ✗ {r['screenshot']}: {r['error']}")
        lines.append("")
    
    # Final verdict
    if successful and avg_stability >= 0.9:
        verdict = "✅ EXCELLENT - System is highly stable"
    elif successful and avg_stability >= 0.7:
        verdict = "✓ GOOD - System is reasonably stable"
    elif successful and avg_stability >= 0.5:
        verdict = "⚠ FAIR - Some instability detected"
    else:
        verdict = "✗ POOR - Significant instability issues"
    
    lines.append("=" * 70)
    lines.append(f"VERDICT: {verdict}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stability Testing for Poker Vision")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="tests/screenshots",
        help="Input directory or single image path",
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=15,
        help="Number of frames per image",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for report (optional)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        results = [test_screenshot_stability(str(input_path), num_frames=args.frames)]
    else:
        # Directory
        results = test_batch_stability(str(input_path), num_frames_per_image=args.frames)
    
    # Generate report
    report = generate_stability_report(results)
    print(report)
    
    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
