#!/bin/bash
# Example commands for running YOLO inference script

# Example 1: Quick test with 10 images using the trained model
echo "Example 1: Testing with 10 images using trained model..."
python scripts/yolo_inference.py \
    --split test \
    --limit 10 \
    --output-dir data/02_processed/example1_model_inference

# Example 2: Process test set with ground truth labels
echo "Example 2: Processing test set with ground truth..."
python scripts/yolo_inference.py \
    --use-ground-truth \
    --split test \
    --output-dir data/02_processed/example2_ground_truth

# Example 3: Process validation set with high confidence threshold
echo "Example 3: Processing validation set with high confidence..."
python scripts/yolo_inference.py \
    --split val \
    --conf-threshold 0.7 \
    --output-dir data/02_processed/example3_high_conf

# Example 4: Generate only visualizations (no crops)
echo "Example 4: Generating visualizations only..."
python scripts/yolo_inference.py \
    --split test \
    --limit 20 \
    --no-crops \
    --output-dir data/02_processed/example4_visualizations

# Example 5: Process all training images with ground truth
echo "Example 5: Processing all training images with ground truth..."
python scripts/yolo_inference.py \
    --use-ground-truth \
    --split train \
    --output-dir data/02_processed/train_crops_ground_truth

# Example 6: Full dataset processing with model
echo "Example 6: Processing full dataset with trained model..."
python scripts/yolo_inference.py \
    --output-dir data/02_processed/full_dataset_crops

echo "All examples completed!"
