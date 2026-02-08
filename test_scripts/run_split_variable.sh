#!/bin/bash

    python3 generate_coop_dataset.py --target_order BGR GBR --num_episodes 40

echo "Starting GBR"
    python3 process_coop_data.py \
    --dataset_dir ./data/variable_coop_dataset/GBR \
    --num_episodes 40 \
    --split_step 280 \
    --phase2_len 290 \
    --overlap_len 100




echo "Starting BGR"
    python3 process_coop_data.py \
    --dataset_dir ./data/variable_coop_dataset/BGR \
    --num_episodes 40 \
    --split_step 280 \
    --phase2_len 290 \
    --overlap_len 100



    python3 visualize_episodes.py --dataset_dir ./data/variable_coop_dataset/GBR_phase1 --episode_idx 0
    python3 visualize_episodes.py --dataset_dir ./data/variable_coop_dataset/GBR_phase2_left --episode_idx 0
    python3 visualize_episodes.py --dataset_dir ./data/variable_coop_dataset/GBR_phase2_right --episode_idx 0
    python3 visualize_episodes.py --dataset_dir ./data/variable_coop_dataset/BGR_phase1 --episode_idx 0
    python3 visualize_episodes.py --dataset_dir ./data/variable_coop_dataset/BGR_phase2_left --episode_idx 0
    python3 visualize_episodes.py --dataset_dir ./data/variable_coop_dataset/BGR_phase2_right --episode_idx 0

echo "All evaluations finished!"
