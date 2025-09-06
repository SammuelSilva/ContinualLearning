# 1. Standard training (baseline)
python scripts/train_hierarchical.py \
    --lora_config attention_only \
    --num_tasks 10 \
    --num_epochs 20

# 2. Hierarchical with TRIM
python scripts/train_hierarchical.py \
    --use_hierarchical \
    --use_unknown_data \
    --include_unknown_test \
    --lora_config attention_only \
    --num_tasks 10 \
    --num_epochs 4 \
    --max_tasks_per_block 3 \
    --min_tasks_to_merge 2 \
    --similarity_threshold 0.7 \
    --trim_percentage 0.3 \
    --max_accuracy_drop 10 \
    --ablation_samples 100 \
    --visualize