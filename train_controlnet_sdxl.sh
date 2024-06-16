module load cuda/12.1 nccl/2.18.3/cuda12.1

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/share0/seanlee/save"
export CUDA_LAUNCH_BLOCKING=1


accelerate launch --config_file ./ac_config.yaml --main_process_port 29501 train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path thibaud/controlnet-openpose-sdxl-1.0 \
 --output_dir=$OUTPUT_DIR \
 --resolution=1024 \
 --train_data_dir='/share0/heywon/jsonl/openImage_final.jsonl' \
 --cache_dir "/share0/seanlee/" \
 --checkpointing_steps 200 \
 --checkpoints_total_limit 10 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision='fp16' \
 --caption_column='text' \
 --conditioning_image_column='condition_image' \
 --use_8bit_adam \
 --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path=thibaud/controlnet-openpose-sdxl-1.0 \
 --num_train_epochs=1 \
 --tracker_project_name cnrl \
 --report_to "wandb"