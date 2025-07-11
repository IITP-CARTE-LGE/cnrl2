#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
import gc
import inspect
import logging
import math
import os
import random
import shutil
from typing import Optional
from contextlib import nullcontext
from pathlib import Path
import copy
from collections import defaultdict

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from reward import RewardComputation

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    # DDPMScheduler,
    DDIMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def log_validation(vae, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
    else:
        controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
        if args.pretrained_vae_model_name_or_path is not None:
            vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, torch_dtype=weight_dtype)
        else:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
            )

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt, image=validation_image, num_inference_steps=20, generator=generator
                ).images[0]
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps when training.",
    )
    parser.add_argument(
        "--train_inference_guidance_scale", #guidance_scale
        type=float,
        default=5.0,
        help="Guidance scale of inference when training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_log_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--project_name", type=str)

    parser.add_argument("--train_clip_range", type=float, default=1e-4, help="Clip range")
    parser.add_argument("--train_adv_clip_max", type=float, default=5, help="Clip advantages to the range")
    parser.add_argument("--reward", type=str, default=None, choices=["keypoints", "feature"])


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset("json", data_files=args.train_data_dir, cache_dir=args.cache_dir)
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompts_example = ['intricate','neutral yet realistic', 'detailed','4k','film','8k','Canon','HD','sharp focus','perfect face', 'perfect hands']
    prompt_embeds_list = []
    
    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            additional_prompts = ', '.join(random.sample(prompts_example, 3))
            captions.append(f"{caption}, {additional_prompts}, realsitic, High-resolution, highly detailed")
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    
    ### assumed negative prompts to be zero_out=True
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    # negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    # negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
    #     bs_embed * num_images_per_prompt, -1
    # )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution)
        ]
    )

    to_tensor_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [Image.open(image).convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]

        condition_images = [Image.open(image).convert("RGB") for image in examples[args.conditioning_image_column]]
        condition_images = [image_transforms(image) for image in condition_images]
        conditioning_images = [to_tensor_transforms(image) for image in condition_images]

        examples["original_image"] = images
        examples["condition_image"] = condition_images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset

def collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    original_image = [example["original_image"] for example in examples]
    condition_image = [example["condition_image"] for example in examples]
    
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompts = [example["prompts"] for example in examples]
    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    ### text embeddings for inference
    neg_prompt_ids = torch.stack([torch.tensor(example["neg_prompt_embeds"]) for example in examples])
    neg_add_text_embeds = torch.stack([torch.tensor(example["neg_text_embeds"]) for example in examples])
    neg_add_time_ids =  torch.stack([torch.tensor(example["neg_time_ids"]) for example in examples])
        
    return {
        "prompts": prompts,
        "original_image": original_image,
        "condition_image": condition_image,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "neg_prompt_ids": neg_prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        "neg_unet_added_conditions": {"text_embeds": neg_add_text_embeds, "time_ids": neg_add_time_ids},
    }

def calculate_loss(latents, timesteps, next_latents, log_probs, advantages, conditioning_pixel_values, prompt_embeds, text_embeds, time_ids):
    """
    Calculate the loss for a batch of an unpacked sample

    Args:
        latents (torch.Tensor):
            The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
        timesteps (torch.Tensor):
            The timesteps sampled from the diffusion model, shape: [batch_size]
        next_latents (torch.Tensor):
            The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
        log_probs (torch.Tensor):
            The log probabilities of the latents, shape: [batch_size]
        advantages (torch.Tensor):
            The advantages of the latents, shape: [batch_size]
        embeds (torch.Tensor):
            The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
            Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

    Returns:
        loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
        (all of these are of shape (1,))
    """


    return loss, approx_kl, clipfrac

def loss_(
    advantages: torch.Tensor,
    clip_range: float,
    ratio: torch.Tensor,
):
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range,
    )
    return torch.mean(torch.maximum(unclipped_loss, clipped_loss))

def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)
        
# def scheduler_step(
#     self,
#     model_output: torch.FloatTensor,
#     timestep: int,
#     sample: torch.FloatTensor,
#     eta: float = 0.0,
#     use_clipped_model_output: bool = False,
#     generator = None,
#     variance_noise = None,
#     return_dict: bool = True,
# ):
#     """
#     Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
#     process from the learned model outputs (most often the predicted noise).

#     Args:
#         model_output (`torch.FloatTensor`):
#             The direct output from learned diffusion model.
#         timestep (`float`):
#             The current discrete timestep in the diffusion chain.
#         sample (`torch.FloatTensor`):
#             A current instance of a sample created by the diffusion process.
#         eta (`float`):
#             The weight of noise for added noise in diffusion step.
#         use_clipped_model_output (`bool`, defaults to `False`):
#             If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
#             because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
#             clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
#             `use_clipped_model_output` has no effect.
#         generator (`torch.Generator`, *optional*):
#             A random number generator.
#         variance_noise (`torch.FloatTensor`):
#             Alternative to generating noise with `generator` by directly providing the noise for the variance
#             itself. Useful for methods such as [`CycleDiffusion`].
#         return_dict (`bool`, *optional*, defaults to `True`):
#             Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

#     Returns:
#         [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
#             If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
#             tuple is returned where the first element is the sample tensor.

#     """
#     if self.num_inference_steps is None:
#         raise ValueError(
#             "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
#         )

#     # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
#     # Ideally, read DDIM paper in-detail understanding

#     # Notation (<variable name> -> <name in paper>
#     # - pred_noise_t -> e_theta(x_t, t)
#     # - pred_original_sample -> f_theta(x_t, t) or x_0
#     # - std_dev_t -> sigma_t
#     # - eta -> η
#     # - pred_sample_direction -> "direction pointing to x_t"
#     # - pred_prev_sample -> "x_t-1"

#     # 1. get previous step value (=t-1)
#     prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

#     # 2. compute alphas, betas
#     self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)
#     alpha_prod_t = self.alphas_cumprod[timestep]
#     alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

#     beta_prod_t = 1 - alpha_prod_t

#     # 3. compute predicted original sample from predicted noise also called
#     # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#     if self.config.prediction_type == "epsilon":
#         pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
#         pred_epsilon = model_output
#     elif self.config.prediction_type == "sample":
#         pred_original_sample = model_output
#         pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
#     elif self.config.prediction_type == "v_prediction":
#         pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
#         pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
#     else:
#         raise ValueError(
#             f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
#             " `v_prediction`"
#         )

#     # 4. Clip or threshold "predicted x_0"
#     if self.config.thresholding:
#         pred_original_sample = self._threshold_sample(pred_original_sample)
#     elif self.config.clip_sample:
#         pred_original_sample = pred_original_sample.clamp(
#             -self.config.clip_sample_range, self.config.clip_sample_range
#         )

#     # 5. compute variance: "sigma_t(η)" -> see formula (16)
#     # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
#     variance = self._get_variance(timestep, prev_timestep)
#     std_dev_t = eta * variance ** (0.5)

#     if use_clipped_model_output:
#         # the pred_epsilon is always re-derived from the clipped x_0 in Glide
#         pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

#     # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#     pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

#     # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#     prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

#     if variance_noise is not None and generator is not None:
#         raise ValueError(
#             "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
#             " `variance_noise` stays `None`."
#         )

#     if variance_noise is None:
#         variance_noise = randn_tensor(
#             model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
#         )
#     variance = std_dev_t * variance_noise

#     prev_sample = prev_sample_mean + variance
        
#     # log prob of prev_sample given prev_sample_mean and std_dev_t
#     log_prob = (
#         -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
#         - torch.log(std_dev_t)
#         - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
#     )
#     # mean along all but batch dimension
#     log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

#     return prev_sample, log_prob

def scheduler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    ):
    """

    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)

    Returns:
        `DDPOSchedulerOutput`: the predicted sample at the previous timestep and the log probability of the sample
    """

    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob

def _left_broadcast(input_tensor, shape):
    """
    As opposed to the default direction of broadcasting (right to left), this function broadcasts
    from left to right
        Args:
            input_tensor (`torch.FloatTensor`): is the tensor to broadcast
            shape (`Tuple[int]`): is the shape to broadcast to
    """
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError(
            "The number of dimensions of the tensor to broadcast cannot be greater than the length of the shape to broadcast to"
        )
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)

def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    num_train_timesteps = int(args.num_train_inference_steps * 1.0) #(config.sample_num_steps ,config.train_timestep_fraction)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps * num_train_timesteps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    reward_computation = RewardComputation(accelerator.device, args.reward)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    
    
    #####TODO: Initialize Reward Model
    ###reward model initialize
    ###reward model initialize
    ###reward model initialize
    #####TODO: Initialize Reward Model
    
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    controlnet.train()

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        ### this is for training
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        
        ### this is for inference
        neg_prompt_embeds = negative_prompt_embeds.to(accelerator.device)
        neg_add_time_ids = add_time_ids.to(accelerator.device)
        neg_add_text_embeds = negative_pooled_prompt_embeds.to(accelerator.device)
        neg_unet_added_cond_kwargs = {"neg_text_embeds": neg_add_text_embeds, "neg_time_ids": neg_add_time_ids}

        return {"prompts": prompt_batch, "prompt_embeds": prompt_embeds, "neg_prompt_embeds": neg_prompt_embeds, **unet_added_cond_kwargs, **neg_unet_added_cond_kwargs}

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )

    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)

    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb":{"name":args.project_name}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar_epoch = tqdm(
        range(0, args.num_train_epochs),
        initial=initial_global_step,
        desc="epochs",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar_step = tqdm(
        range(0, args.max_train_steps * num_train_timesteps),
        initial=initial_global_step,
        desc="global_step",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            #####TODO[Done]: we generate images, latents, log_probs
            samples = []
            prompt_image_pairs = []
            
            controlnet.eval()

            with torch.no_grad():
                num_channels_latents = unet.config.in_channels
                shape = (
                    args.train_batch_size,
                    num_channels_latents,
                    unet.config.sample_size,
                    unet.config.sample_size
                )
                
                ### preparing latents
                latents = randn_tensor(shape, device=accelerator.device, dtype=weight_dtype) #add generator if needed, device needs to be checked
                latents = latents * noise_scheduler.init_noise_sigma
                all_latents = [latents]
                all_log_probs = []
                
                ###preparing timesteps
                noise_scheduler.set_timesteps(args.num_train_inference_steps, device=latents.device)
                timesteps = noise_scheduler.timesteps        
                extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, None, 1.0) #scheduler, generator, eta
                
                ### for cond_scale
                controlnet_keep = []
                for i in range(len(timesteps)):
                    keeps = [
                        1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                        for s, e in zip([0.0], [1.0])
                    ]
                    controlnet_keep.append(keeps[0])
                
                
                is_unet_compiled = is_compiled_module(unet)
                is_controlnet_compiled = is_compiled_module(controlnet)
                is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
                
                ### preparing encoder_hidden_states, controlnet_cond, added_cond_kwargs from dataloader
                inf_encoder_hidden_states = torch.cat([batch["neg_prompt_ids"], batch["prompt_ids"]], dim=0)
                inf_controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                inf_add_text_embeds = torch.cat([batch['neg_unet_added_conditions']['text_embeds'], batch['unet_added_conditions']['text_embeds']], dim=0)
                inf_add_time_ids = torch.cat([batch['neg_unet_added_conditions']['time_ids'], batch['unet_added_conditions']['time_ids']], dim=0)
                inf_added_cond_kwargs = {"text_embeds": inf_add_text_embeds, "time_ids": inf_add_time_ids}
                
                ###generation
                # (1) saving all latents & log_probs is needed
                # (2) do_classifier_free_guidance=True : make it double
                for i, t in enumerate(timesteps):
                    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    control_model_input = latent_model_input
                
                    controlnet_cond_scale = 1.0
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    # ControlNet conditioning.
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        control_model_input, #매번 오는 거
                        t, #매번 오는 거
                        encoder_hidden_states=inf_encoder_hidden_states, #
                        controlnet_cond=torch.cat([inf_controlnet_image] * 2),
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        added_cond_kwargs=inf_added_cond_kwargs,
                        return_dict=False,
                    )

                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=inf_encoder_hidden_states,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=inf_added_cond_kwargs,
                        down_block_additional_residuals=[
                            d_sample.to(dtype=weight_dtype) for d_sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                    
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.train_inference_guidance_scale * (noise_pred_text - noise_pred_uncond)

                    ### TODO[Done]:we need latents, and log_probs
                    # latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    latents, log_probs = scheduler_step(noise_scheduler, noise_pred, t, latents, **extra_step_kwargs)
                    
                    all_latents.append(latents)
                    all_log_probs.append(log_probs)
                    ### TODO[Done]:we need latents, and log_probs

                ### latents to image
                needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
                if needs_upcasting:
                    upcast_vae(vae)
                    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)

                has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
                has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents = latents * latents_std / vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / vae.config.scaling_factor

                image = vae.decode(latents, return_dict=False)[0]

                # cast back to fp16 if needed
                if needs_upcasting:
                    vae.to(dtype=torch.float16)
                    
                images = image_processor.postprocess(image, output_type='pil')
                #####TODO[Done]: we generate images, latents, log_probs


            ### format change for the reward part
            latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = noise_scheduler.timesteps.repeat(args.train_batch_size, 1)  # (batch_size, num_steps)

            samples.append(
                {
                    "prompts": batch["prompts"],
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "timesteps": timesteps,
                    "conditioning_pixel_values": batch["conditioning_pixel_values"],
                    "prompt_ids": batch["prompt_ids"], 
                    "text_embeds": batch['unet_added_conditions']["text_embeds"],
                    "time_ids": batch['unet_added_conditions']["time_ids"],
                    "neg_prompt_ids": batch["neg_prompt_ids"], 
                    "neg_text_embeds": batch['neg_unet_added_conditions']["text_embeds"],
                    "neg_time_ids": batch['neg_unet_added_conditions']["time_ids"],
                    "log_probs": log_probs,
                }
            )

            prompt_image_pairs.append([batch["original_image"], batch["condition_image"], images, batch["prompts"], {}])
            torch.cuda.empty_cache()

                    # ### TODO:we need latents, and log_probs
                    # latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


            #####TODO: we generate images, latents, log_probs
            
            #####TODO (DONE): we compute rewards
            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)            
            samples = {k: torch.cat([s[k] for s in samples]) if k != "prompts" else [s[k] for s in samples] for k in samples[0].keys()}
            if args.reward == 'feature':
                rewards, rewards_metadata = reward_computation.compute_rewards(prompt_image_pairs, global_step, args.image_log_dir)
            elif args.reward == 'keypoints':
                rewards, rewards_metadata = reward_computation.compute_rewards2(prompt_image_pairs, global_step, args.image_log_dir)
            #####TODO: we compute rewards
            
            #####TODO: turn rewards into advantages
            for i, image_data in enumerate(prompt_image_pairs):
                image_data.extend([rewards[i], rewards_metadata[i]])

            # if self.image_samples_callback is not None:
            #     self.image_samples_callback(prompt_image_data, global_step, self.accelerator.trackers[0])
            rewards = torch.cat(rewards)
            rewards = accelerator.gather(rewards).cpu().numpy()


            accelerator.log(
                {
                    "reward": rewards,
                    "epoch": epoch,
                    "reward_mean": rewards.mean(),
                    "reward_std": rewards.std(),
                },
                step=global_step,
            )
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # ungather advantages;  keep the entries corresponding to the samples on this process
            samples["advantages"] = (
                torch.as_tensor(advantages)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )

            # TODO check here
            del samples["prompts"]

            total_batch_size_, num_timesteps = samples["timesteps"].shape

            ##########TODO HERE TO BE MODIFIED : for loop removed from here
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size_, device=accelerator.device)

            _samples = {}
            for k, v in samples.items():
                v = v.to(accelerator.device)
                _samples[k] = v[perm]
            samples = _samples

            # shuffle along time dimension independently for each sample
            # still trying to understand the code below
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size_)]
            )

            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size_, device=accelerator.device)[:, None],
                    perms,
                ]

            original_keys = samples.keys()
            original_values = samples.values()
            # rebatch them as user defined train_batch_size is different from sample_batch_size
            reshaped_values = [v.reshape(-1, args.train_batch_size, *v.shape[1:]) for v in original_values]

            # Transpose the list of original values
            transposed_values = zip(*reshaped_values)
            samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]
            #####TODO: turn rewards into advantages
                            
                            
            #####TODO: we calculate loss
            controlnet.train()
            info = defaultdict(list)

            for _i, sample in enumerate(samples_batched):
                for j in range(num_train_timesteps//2):
                    with accelerator.accumulate(controlnet):
                        latents = sample["latents"][:, j] 
                        timesteps = sample["timesteps"][:, j]
                        next_latents = sample["next_latents"][:, j]
                        log_probs = sample["log_probs"][:, j]
                        advantages = sample["advantages"]
                        conditioning_pixel_values = sample["conditioning_pixel_values"].to(dtype=weight_dtype)
                        prompt_ids = sample["prompt_ids"]
                        text_embeds = sample["text_embeds"]
                        time_ids = sample["time_ids"]
                        neg_prompt_ids = sample["neg_prompt_ids"]
                        neg_text_embeds = sample["neg_text_embeds"]
                        neg_time_ids = sample["neg_time_ids"]

                        encoder_hidden_states = torch.cat([neg_prompt_ids, prompt_ids], dim=0)
                        add_text_embeds = torch.cat([neg_text_embeds, text_embeds], dim=0)
                        add_time_ids = torch.cat([neg_time_ids, time_ids], dim=0)


                        with accelerator.autocast():
                            controlnet_image = conditioning_pixel_values
                            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                            down_block_res_samples, mid_block_res_sample = controlnet(
                                torch.cat([latents] * 2), #매번 오는 거
                                torch.cat([timesteps] * 2), #매번 오는 거
                                encoder_hidden_states=encoder_hidden_states, #
                                controlnet_cond=torch.cat([controlnet_image] * 2),
                                conditioning_scale=cond_scale,
                                guess_mode=False,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )

                            noise_pred = unet(
                                torch.cat([latents] * 2),
                                torch.cat([timesteps] * 2),
                                encoder_hidden_states=encoder_hidden_states,
                                timestep_cond=None,
                                cross_attention_kwargs=None,
                                added_cond_kwargs=added_cond_kwargs,
                                down_block_additional_residuals=[
                                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                                ],
                                mid_block_additional_residual=mid_block_res_sample,
                                return_dict=False,
                            )[0]

                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + args.train_inference_guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # TODO: Change
                            _, log_prob = scheduler_step(noise_scheduler, noise_pred, timesteps, latents, **extra_step_kwargs, prev_sample=next_latents)
                            
                        advantages = torch.clamp(
                            advantages,
                            -args.train_adv_clip_max,
                            args.train_adv_clip_max,
                        )

                        ratio = torch.exp(log_prob - log_probs)

                        loss = loss_(advantages, args.train_clip_range, ratio)
                        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

                        clipfrac = torch.mean((torch.abs(ratio - 1.0) > args.train_clip_range).float())


                        info["approx_kl"].append(approx_kl)
                        info["clipfrac"].append(clipfrac)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        # changed this ----- trainable_layers (deleted)
                        if accelerator.sync_gradients:
                            params_to_clip = controlnet.parameters()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    ################# TODO: need to check here
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch})
                        accelerator.log(info, step=global_step)
                        progress_bar_step.update(1)
                        global_step += 1
                        info = defaultdict(list)

                #####TODO: we calculate loss

    ###########Newly added codes end here###########
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if step != 0 and step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit and accelerator.is_main_process:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        accelerator.wait_for_everyone() 
                        if accelerator.is_main_process:
                            controlnet_ = unwrap_model(controlnet)
                            controlnet_.save_pretrained(save_path)

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae=vae,
                            unet=unet,
                            controlnet=controlnet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            # progress_bar.set_postfix(**logs)

            # if global_step >= args.max_train_steps:
            #     break
        if accelerator.sync_gradients:
            progress_bar_epoch.update(1)
        # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of validation.
        # Setting `vae`, `unet`, and `controlnet` to None to load automatically from `args.output_dir`.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                vae=None,
                unet=None,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)