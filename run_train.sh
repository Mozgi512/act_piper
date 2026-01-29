#!/bin/bash

# Train1
echo "Starting training1..."
python3 imitate_episodes.py \
    --task_name sim_coop_scripted \
    --ckpt_dir ckpt/cooperation_150eps \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 40000 --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 --num_episodes 150


echo "Starting evaluation for cooperation_150eps..."
python3 imitate_episodes.py \
    --task_name sim_coop_scripted \
    --ckpt_dir ckpt/cooperation_150eps \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 40000 \
    --lr 1e-5 \
    --seed 0 \
    --eval \
    --eval_interval 1000 \
    --num_rollouts 50 \
    --no_video

echo "Starting training2..."
python3 imitate_episodes.py \
    --task_name sim_coop_scripted \
    --ckpt_dir ckpt/cooperation_200eps \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 40000 --lr 1e-5 \
    --seed 0 \
    --num_workers 4 \
    --persistent_workers \
    --prefetch_factor 2 \
    --use_cache \
    --use_cuda_graph \
    --validation_interval 100 --num_episodes 200
    
echo "Starting evaluation for cooperation_200eps..."
python3 imitate_episodes.py \
    --task_name sim_coop_scripted \
    --ckpt_dir ckpt/cooperation_200eps \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 40000 \
    --lr 1e-5 \
    --seed 0 \
    --eval \
    --eval_interval 1000 \
    --num_rollouts 50 \
    --no_video

echo "All training jobs finished!"
