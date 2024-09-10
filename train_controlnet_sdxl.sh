export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/share0/heywon/test"


accelerate launch --config_file /home/heywon/cnrl2/ac_config.yaml --main_process_port 29501 /home/heywon/cnrl2/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path thibaud/controlnet-openpose-sdxl-1.0 \
 --output_dir=$OUTPUT_DIR \
 --resolution=1024 \
 --train_data_dir='/share0/heywon/temp/condition_train_0615.jsonl' \
 --cache_dir /share0/heywon/cache \
 --checkpointing_steps 50 \
 --checkpoints_total_limit 15 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --learning_rate 1e-6 \
 --num_train_inference_steps 50 \
 --image_log_dir /share0/heywon/cache/test \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision='fp16' \
 --caption_column='text' \
 --conditioning_image_column='condition_image' \
 --image_column='image'\
 --use_8bit_adam \
 --num_train_epochs=3 \
#  --tracker_project_name cnrl6 \
#  --report_to "wandb" \