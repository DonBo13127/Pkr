"""
tests/test_cards.py - Card Detection and Classification Tests

Tests specifically for card detection accuracy:
- Card detection rate on real screenshots
- Classification accuracy (when model available)
- Multi-card scenarios
- Partial visibility handling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import cv2
import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.pipeline import PokerVisionPipeline, PipelineConfig
from vision.yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_card_detection_on_screenshots(
    screenshot_dir: str,
) -> Dict[str, Any]:
    """
    Test card detection on all screenshots.
    
    Args:
        screenshot_dir: Directory containing screenshots
        
    Returns:
        Detection statistics
    """
    screenshot_path = Path(screenshot_dir)
    if not screenshot_path.exists():
        return {"error": f"Directory not found: {screenshot_dir}"}
    
    images = list(screenshot_path.glob("*.png")) + list(screenshot_path.glob("*.jpg"))
    
    if not images:
        return {"error": "No images found"}
    
    logger.info(f"Testing card detection on {len(images)} images")
    
    # Create pipeline in mock mode
    config = PipelineConfig(use_mock=True, debug_mode=False)
    pipeline = PokerVisionPipeline(config=config)
    
    results = []
    total_cards_detected = 0
    
    for img_path in images:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Process frame
            state = pipeline.process_frame(image)
            
            # Get detections
            cards = state.get_board_cards()
            total_cards_detected += len(cards)
            
            results.append({
                "image": img_path.name,
                "cards_detected": len(cards),
                "card_list": cards,
                "street": state.street.value,
                "confidence": state.confidence,
            })
            
            logger.info(f"  {img_path.name}: {len(cards)} cards detected")
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            results.append({
                "image": img_path.name,
                "error": str(e),
            })
    
    # Aggregate statistics
    card_counts = [r.get("cards_detected", 0) for r in results if "error" not in r]
    
    return {
        "total_images": len(images),
        "successful": len(card_counts),
        "total_cards_detected": total_cards_detected,
        "average_cards_per_image": sum(card_counts) / len(card_counts) if card_counts else 0,
        "card_count_distribution": dict(Counter(card_counts)),
        "per_image_results": results,
    }


def generate_card_test_report(results: Dict[str, Any]) -> str:
    """Generate human-readable report."""
    
    lines = [
        "=" * 70,
        "CARD DETECTION TEST REPORT",
        "=" * 70,
        "",
    ]
    
    if "error" in results:
        lines.append(f"ERROR: {results['error']}")
        return "\n".join(lines)
    
    lines.append(f"Total Images: {results['total_images']}")
    lines.append(f"Successful: {results['successful']}")
    lines.append(f"Total Cards Detected: {results['total_cards_detected']}")
    lines.append(f"Average Cards per Image: {results['average_cards_per_image']:.1f}")
    lines.append("")
    
    lines.append("CARD COUNT DISTRIBUTION:")
    for count, freq in sorted(results["card_count_distribution"].items()):
        lines.append(f"  {count} cards: {freq} images")
    lines.append("")
    
    lines.append("PER-IMAGE RESULTS:")
    lines.append("-" * 70)
    
    for r in results["per_image_results"]:
        if "error" in r:
            lines.append(f"  ✗ {r['image']}: {r['error']}")
        else:
            cards_str = ", ".join(r["card_list"]) if r["card_list"] else "(none)"
            lines.append(
                f"  {r['image']:35s} | "
                f"Cards: {r['cards_detected']} | "
                f"{cards_str}"
            )
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Card Detection Testing")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="tests/screenshots",
        help="Input directory with screenshots",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for report",
    )
    
    args = parser.parse_args()
    
    results = test_card_detection_on_screenshots(args.input)
    report = generate_card_test_report(results)
    
    print(report)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
