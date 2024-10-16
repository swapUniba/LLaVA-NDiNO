#!/bin/bash

# useful for debugging/determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

LLM_VERSION="models/ft-only-llama-ita"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="models/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION="llava_llama_3"

NEW_RUN_NAME="ft-long-llama-ita"

accelerate launch --config_file configs/${DISTRIBUTED_CONFIG}_config.yaml --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_processes $(( $GPU_PER_NODE * $COUNT_NODE)) --num_machines $COUNT_NODE --machine_rank $SLURM_PROCID  \
    train.py \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path=./data/conv_dataset.json \
    --image_folder ./data/conv_images \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir "models/${NEW_RUN_NAME}" \
    --max_steps 500 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True