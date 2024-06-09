# export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./save"

accelerate launch --config_file ./ac_config.yaml --main_process_port 29501 train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
 --controlnet_model_name_or_path thibaud/controlnet-openpose-sdxl-1.0 \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir='/scratch/heywon/data/output1.jsonl' \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --caption_column 'text' \
 --image_column 'image' \
 --conditioning_image_column 'condition_image' \
 --enable_xformers_memory_efficient_attention \
 --use_8bit_adam \
#  --seed=42 \

#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --resolution=512 \
#  --train_data_dir='/scratch/heywon/data/output.jsonl' \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --enable_xformers_memory_efficient_attention \
#  --use_8bit_adam \
#  --set_grads_to_none \
#  --mixed_precision fp16 \
#  --caption_column 'text' \
#  --image_column 'image' \
#  --conditioning_image_column 'condition_image'