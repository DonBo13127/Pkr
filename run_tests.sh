#!/bin/bash
# Poker Vision System - Test Runner Script
# Run all tests and generate reports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "POKER VISION SYSTEM - TEST SUITE"
echo "=============================================="
echo ""

# Create output directories
mkdir -p logs output

# Timestamp for reports
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Running main pipeline tests..."
python -m vision.test_pipeline 2>&1 | tee logs/pipeline_test_${TIMESTAMP}.log

echo ""
echo "[$(date)] Running stability tests..."
python tests/test_stability.py \
    --input tests/screenshots \
    --frames 10 \
    --output logs/stability_report_${TIMESTAMP}.txt \
    2>&1 | tee logs/stability_test_${TIMESTAMP}.log

echo ""
echo "[$(date)] Running card detection tests..."
python tests/test_cards.py \
    --input tests/screenshots \
    --output logs/card_report_${TIMESTAMP}.txt \
    2>&1 | tee logs/card_test_${TIMESTAMP}.log

echo ""
echo "=============================================="
echo "ALL TESTS COMPLETED"
echo "=============================================="
echo ""
echo "Reports saved to:"
echo "  - logs/pipeline_test_${TIMESTAMP}.log"
echo "  - logs/stability_report_${TIMESTAMP}.txt"
echo "  - logs/card_report_${TIMESTAMP}.txt"
echo ""
echo "Visualizations saved to: output/"
