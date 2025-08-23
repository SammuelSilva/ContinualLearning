# 1. Standard training (baseline)
python scripts/train_hierarchical.py \
    --lora_config attention_only \
    --num_tasks 10 \
    --num_epochs 20

# 2. Hierarchical with orthogonal merging
python scripts/train_hierarchical.py \
    --use_hierarchical \
    --use_orthogonal_merge \
    --tasks_per_block 2 \
    --merge_strategy qr \
    --lora_config attention_only \
    --visualize

# 3. Adaptive merging strategy
python scripts/train_hierarchical.py \
    --use_hierarchical \
    --adaptive_merge \
    --use_orthogonal_merge \
    --visualize \
    --create_animation

# 4. Progressive merge scheduling
python scripts/train_hierarchical.py \
    --use_hierarchical \
    --progressive_merge \
    --use_orthogonal_merge \
    --initial_block_size 3

# 5. Full experiment with all features
python scripts/train_hierarchical.py \
    --use_hierarchical \
    --use_orthogonal_merge \
    --adaptive_merge \
    --tasks_per_block 2 \
    --lora_config both \
    --lora_rank 8 \
    --lambda_task_unknown 0.5 \
    --lambda_block_unknown 0.3 \
    --visualize \
    --create_animation \
    --num_epochs 30 \
    --experiment_name "full_hierarchical_experiment"

# 6. Ablation studies
python scripts/train_hierarchical.py \
    --ablation_mode different_ranks \
    --use_hierarchical

python scripts/train_hierarchical.py \
    --ablation_mode different_blocks \
    --use_hierarchical

python scripts/train_hierarchical.py \
    --ablation_mode no_orthogonal \
    --use_hierarchical

# 7. Resume training
python scripts/train_hierarchical.py \
    --resume \
    --resume_dir ./results/hierarchical_attention_only_20240101_120000 \
    --use_hierarchical

# 8. Evaluation only
python scripts/train_hierarchical.py \
    --eval_only \
    --eval_checkpoint ./checkpoints/best_model.pt \
    --use_hierarchical