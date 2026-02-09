#!/bin/bash
# Complete Training Pipeline: Pre-training + Fine-tuning (Sequential)
# Runs all 6 jobs sequentially to avoid VRAM overflow

set -e  # Exit on error

echo "=========================================="
echo "Complete Training Pipeline"
echo "=========================================="
echo ""
echo "This script will run:"
echo "  Phase 1: 3 pre-training jobs"
echo "  Phase 2: 3 fine-tuning jobs"
echo ""
echo "Each job takes ~20-30 hours."
echo "Total time: ~120-180 hours (5-7 days)"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Create all directories
mkdir -p ckpt3/pretrained_left
mkdir -p ckpt3/pretrained_right
mkdir -p ckpt3/pretrained_c
mkdir -p ckpt3/finetune_left
mkdir -p ckpt3/finetune_right
mkdir -p ckpt3/finetune_c
mkdir -p logs

echo ""
echo "=========================================="
echo "PHASE 1: PRE-TRAINING"
echo "=========================================="
echo ""

# ============================================
echo "=========================================="
echo "[2/6] Pre-training Right Arm"
echo "=========================================="
echo ""

python3 independent_imitate_episodes.py \
    --task_name sim_dataset_i \
    --ckpt_dir ckpt3/pretrained_right \
    --dataset_dir data3/pretrain_right_padded \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 20000 --lr 1e-5 \
    --seed 0 --arm right --num_episodes 100 --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 --episode_len 370 \
    2>&1 | tee logs/pretrain_right.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ [2/6] pretrained_right failed!"
    exit 1
fi

echo ""
echo "✅ [2/6] pretrained_right completed!"
echo ""

# ============================================
# Pre-training 3/3: Cooperative
# ============================================
echo "=========================================="
echo "[3/6] Pre-training Cooperative"
echo "=========================================="
echo ""

python3 imitate_episodes.py \
    --task_name sim_dataset_c \
    --ckpt_dir ckpt3/pretrained_c \
    --dataset_dir data3/pretrain_c_padded \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 20000 --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 --num_episodes 100 --episode_len 770 \
    2>&1 | tee logs/pretrain_c.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ [3/6] pretrained_c failed!"
    exit 1
fi

echo ""
echo "✅ [3/6] pretrained_c completed!"
echo ""

echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo ""
echo "Verifying pre-training checkpoints..."
ls -lh ckpt3/pretrained_*/policy_best.ckpt
echo ""

echo ""
echo "=========================================="
echo "PHASE 2: FINE-TUNING"
echo "=========================================="
echo ""

# ============================================
# Fine-tuning 1/3: Left Arm
# ============================================
echo "=========================================="
echo "[4/6] Fine-tuning Left Arm"
echo "=========================================="
echo ""

python3 independent_imitate_episodes.py \
    --task_name sim_dataset_i \
    --ckpt_dir ckpt3/finetune_left \
    --dataset_dir data3/L_padded \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 20000 --lr 1e-5 \
    --seed 0 --arm left --num_episodes 210 --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --load_ckpt ckpt3/pretrained_left/policy_best.ckpt \
    --validation_interval 100 --episode_len 390 --reset_optimizer \
    2>&1 | tee logs/finetune_left.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ [4/6] finetune_left failed!"
    exit 1
fi

echo ""
echo "✅ [4/6] finetune_left completed!"
echo ""

# ============================================
# Fine-tuning 2/3: Right Arm
# ============================================
echo "=========================================="
echo "[5/6] Fine-tuning Right Arm"
echo "=========================================="
echo ""

python3 independent_imitate_episodes.py \
    --task_name sim_dataset_i \
    --ckpt_dir ckpt3/finetune_right \
    --dataset_dir data3/R_padded \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 20000 --lr 1e-5 \
    --seed 0 --arm right --num_episodes 319 --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --load_ckpt ckpt3/pretrained_right/policy_best.ckpt \
    --validation_interval 100 --episode_len 390 --reset_optimizer \
    2>&1 | tee logs/finetune_right.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ [5/6] finetune_right failed!"
    exit 1
fi

echo ""
echo "✅ [5/6] finetune_right completed!"
echo ""

# ============================================
# Fine-tuning 3/3: Cooperative
# ============================================
echo "=========================================="
echo "[6/6] Fine-tuning Cooperative"
echo "=========================================="
echo ""

python3 imitate_episodes.py \
    --task_name sim_dataset_c \
    --ckpt_dir ckpt3/finetune_c \
    --dataset_dir data3/C_padded \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 20000 --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --load_ckpt ckpt3/pretrained_c/policy_best.ckpt \
    --validation_interval 100 --num_episodes 180 --episode_len 791 --reset_optimizer \
    2>&1 | tee logs/finetune_c.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ [6/6] finetune_c failed!"
    exit 1
fi

echo ""
echo "✅ [6/6] finetune_c completed!"
echo ""

echo "=========================================="
echo "Phase 2 Complete!"
echo "=========================================="
echo ""
echo "Verifying fine-tuning checkpoints..."
ls -lh ckpt3/finetune_*/policy_best.ckpt
echo ""

echo ""
echo "=========================================="
echo "🎉 TRAINING PIPELINE COMPLETE! 🎉"
echo "=========================================="
echo ""
echo "All 6 jobs completed successfully:"
echo "  ✅ pretrained_left"
echo "  ✅ pretrained_right"
echo "  ✅ pretrained_c"
echo "  ✅ finetune_left"
echo "  ✅ finetune_right"
echo "  ✅ finetune_c"
echo ""
echo "Final checkpoints:"
ls -lh ckpt3/*/policy_best.ckpt
echo ""
echo "Logs saved in: logs/"
echo ""
