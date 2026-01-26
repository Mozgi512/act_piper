#!/bin/bash

# Train1
echo "Starting training for phase1..."
python3 imitate_episodes.py \
    --task_name sim_coop_phase1_scripted \
    --ckpt_dir ckpt/cooperation/_phase1 \
    --dataset_dir piper_scripted_dataset/cooperation/_phase1 \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 20000 \
    --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 \
    --num_episodes 100

# Train2
echo "Starting training for phase2_left..."
python3 independent_imitate_episodes.py \
    --task_name sim_coop_phase2_left_scripted \
    --ckpt_dir ckpt/cooperation/_phase2_left \
    --dataset_dir piper_scripted_dataset/cooperation/_phase2_left \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 20000 \
    --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 \
    --num_episodes 100 \
    --arm left

# Train3
echo "Starting training for phase2_right..."
python3 independent_imitate_episodes.py \
    --task_name sim_coop_phase2_right_scripted \
    --ckpt_dir ckpt/cooperation/_phase2_right \
    --dataset_dir piper_scripted_dataset/cooperation/_phase2_right \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 20000 \
    --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 \
    --num_episodes 100 \
    --arm right

echo "All training jobs finished!"
