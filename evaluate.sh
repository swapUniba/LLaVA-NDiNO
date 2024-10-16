#!/bin/bash

# evaluate mBlip
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m lmms_eval --model mblip --model_args pretrained="Gregor/mblip-mt0-xl,dtype=bfloat16" --tasks gqa-it --batch_size 1 --log_samples --log_samples_suffix mblip_it --output_path ./logs-mblip

# evaluate LLaVA-NDiNO
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m lmms_eval --model llava_hf --model_args pretrained="swap-uniba/LLaVA-NDiNO_pt_short_it,dtype=bfloat16,device_map=auto" --tasks gqa-it --batch_size 1 --log_samples --log_samples_suffix llava_it --output_path ./logs-llava-pt-it
