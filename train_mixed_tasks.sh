#!/bin/bash
# Training script for mixed task dataset (III, CCC, IIC)
# Make sure you have prepared and padded your dataset first!

set -e  # Exit on error

# Configuration
TASK_NAME="sim_many_cubes"
DATASET_DIR="data/training"  # Your padded dataset directory
CKPT_DIR="ckpt/mixed_tasks"
NUM_EPOCHS=40000
BATCH_SIZE=32
VALIDATION_INTERVAL=100

echo "========================================="
echo "Training ACT on Mixed Tasks Dataset"
echo "========================================="
echo "Dataset: $DATASET_DIR"
echo "Checkpoint: $CKPT_DIR"
echo "========================================="

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory $DATASET_DIR not found!"
    echo "Please run the dataset preparation workflow first:"
    echo "  1. python generate_dataset.py ..."
    echo "  2. python merge_datasets.py ..."
    echo "  3. python pad_dataset.py ..."
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

# Count episodes
NUM_EPISODES=$(ls -1 "$DATASET_DIR"/episode_*.hdf5 2>/dev/null | wc -l)
echo "Found $NUM_EPISODES episodes in dataset"

if [ "$NUM_EPISODES" -eq 0 ]; then
    echo "Error: No episode files found in $DATASET_DIR"
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Training with performance optimizations
CUDA_LAUNCH_BLOCKING=0 TORCH_CUDNN_V8_API_ENABLED=1 \
python3 imitate_episodes.py \
    --task_name "$TASK_NAME" \
    --ckpt_dir "$CKPT_DIR" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size "$BATCH_SIZE" \
    --dim_feedforward 3200 \
    --num_epochs "$NUM_EPOCHS" \
    --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --validation_interval "$VALIDATION_INTERVAL" \
    --num_episodes "$NUM_EPISODES"

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Checkpoint saved to: $CKPT_DIR"
echo "Evaluation results: $CKPT_DIR/evaluation_results.csv"
echo ""
echo "To evaluate the trained model:"
echo "  python3 imitate_episodes.py \\"
echo "    --task_name $TASK_NAME \\"
echo "    --ckpt_dir $CKPT_DIR \\"
echo "    --policy_class ACT \\"
echo "    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 \\"
echo "    --eval --onscreen_render --num_rollouts 50"
echo ""
echo "To plot training curves:"
echo "  python3 plot_evaluation.py \\"
echo "    --csv_path $CKPT_DIR/evaluation_results.csv \\"
echo "    --output_path results/training_curve.png"
echo "========================================="
