export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/heywon/cnrl/save"

accelerate launch --config_file ./ac_config.yaml /home/heywon/cnrl2/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --train_data_dir='/scratch/heywon/data/output.jsonl' \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision='fp16' \
 --image_column='image'\
 --caption_column='text' \
 --conditioning_image_column='condition_image'