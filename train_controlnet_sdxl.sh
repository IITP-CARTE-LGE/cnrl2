export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/share0/heywon/0615"


accelerate launch --config_file ./ac_config.yaml --main_process_port 29501 train_cnrl_pose_cos_reward.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path thibaud/controlnet-openpose-sdxl-1.0 \
 --output_dir=$OUTPUT_DIR \
 --resolution=1024 \
 --train_data_dir='/share0/heywon/temp/condition_train_0615.jsonl' \
 --cache_dir "/share0/heywon/cache" \
 --checkpointing_steps 200 \
 --checkpoints_total_limit 10 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision='fp16' \
 --caption_column='text' \
 --conditioning_image_column='condition_image' \
 --image_column='image'\
 --use_8bit_adam \
 --num_train_epochs=1 \
 --image_log_dir='/share0/heywon/cache/0615'
#  --tracker_project_name cnrl5 \
#  --report_to "wandb"