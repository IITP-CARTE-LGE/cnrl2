#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
 
# conda 환경 활성화.
source  ~/.bashrc
source /scratch2/heywon/train_venv/bin/activate
 
# # cuda 12.1 환경 구성.
# ml purge
# module load cuda/12.1 nccl/2.18.3/cuda12.1

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/scratch2/heywon/cache/feature_openpose1"


accelerate launch --config_file /scratch2/heywon/cnrl2/ac_config.yaml --main_process_port 29501 /home/heywon/cnrl2/train_cnrl_pose_cos_reward.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
 --controlnet_model_name_or_path='thibaud/controlnet-openpose-sdxl-1.0' \
 --output_dir=$OUTPUT_DIR \
 --resolution=1024 \
 --train_data_dir='/scratch2/heywon/cache/one_person_openpose.jsonl' \
 --checkpointing_steps 40 \
 --checkpoints_total_limit 100 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --learning_rate 1e-6 \
 --num_train_inference_steps 50 \
 --image_log_dir='/scratch2/heywon/cache/images' \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision='fp16' \
 --caption_column='text' \
 --conditioning_image_column='condition_image' \
 --reward='feature' \
 --image_column='image'\
 --use_8bit_adam \
 --num_train_epochs=3 \
 --tracker_project_name feature_openpose \
 --report_to "wandb" \
 --project_name feature_openpose1

 
echo "### END DATE=$(date)"
echo "###"