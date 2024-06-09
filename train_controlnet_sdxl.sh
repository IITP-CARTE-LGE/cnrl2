export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./save"

accelerate launch --config_file ./ac_config.yaml --main_process_port 10000 train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path thibaud/controlnet-openpose-sdxl-1.0 \
 --output_dir=$OUTPUT_DIR \
 --resolution=1024 \
 --train_data_dir='/scratch/heywon/data/output.jsonl' \
 --train_batch_size=2 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision='fp16' \
 --image_column='image'\
 --caption_column='text' \
 --conditioning_image_column='condition_image' \
 --use_8bit_adam \
 --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path=thibaud/controlnet-openpose-sdxl-1.0
