#!/bin/bash

# Run evaluation for cooperation_50eps
echo "Starting evaluation for cooperation_50eps..."
python3 imitate_episodes.py \
    --task_name sim_coop_scripted \
    --ckpt_dir ckpt/cooperation_50eps_2 \
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
    --no_video \
    --start_epoch 16000

# Run evaluation for cooperation_75eps
echo "Starting evaluation for cooperation_75eps..."
python3 imitate_episodes.py \
    --task_name sim_coop_scripted \
    --ckpt_dir ckpt/cooperation_75eps_2 \
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

echo "All evaluations finished!"
