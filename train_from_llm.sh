#!/bin/bash

# useful for debugging/determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

LLM_VERSION="models/Meta-Llama-3-8B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="models/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION="plain"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

NEW_RUN_NAME="adapt-ita-llama"

accelerate launch --config_file configs/${DISTRIBUTED_CONFIG}_config.yaml --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_processes $(( $GPU_PER_NODE * $COUNT_NODE)) --num_machines $COUNT_NODE --machine_rank $SLURM_PROCID  \
    train.py \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path=./data/adapt.json \
    --image_folder ./data/adapt_images \
    --pretrain_mm_mlp_adapter="models/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
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
