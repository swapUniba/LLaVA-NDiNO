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
# limitations under the License.

"""Convert LLaVa-NeXT (LLaVa-1.6) checkpoints from the original repository.

URL: https://github.com/haotian-liu/LLaVA/tree/main.


The command used to obtain original logits is the following:
python llava/eval/run_llava.py --model-path "liuhaotian/llava-v1.6-mistral-7b" --image-file "images/llava_v1_5_radar.jpg" --query "What is shown in this image?" --max_new_tokens 100 --temperature 0

Note: logits are tested with torch==2.1.2.
"""

import glob
import json
import argparse
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}


def load_original_state_dict(model_id):

    original_state_dict = {}
    for path in glob.glob(f"{model_id}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def convert_llava_to_hf(model_id, pytorch_dump_folder_path):

    # read json
    with open(f'{model_id}/config.json') as f:
        data = json.load(f)
        print(data)

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        text_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        image_token_index = 32000
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        text_model_id = "lmsys/vicuna-7b-v1.5"
        image_token_index = 32000
    elif model_id == "liuhaotian/llava-v1.6-vicuna-13b":
        text_model_id = "lmsys/vicuna-13b-v1.5"
        image_token_index = 32000
    elif model_id == "liuhaotian/llava-v1.6-34b":
        text_model_id = "NousResearch/Nous-Hermes-2-Yi-34B"
        image_token_index = 64000
    elif model_id == "lmms-lab/llama3-llava-next-8b":
        text_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        image_token_index = 128256
    elif model_id == "lmms-lab/llava-next-72b":
        text_model_id = "Qwen/Qwen1.5-72B-Chat"
        image_token_index = 151646
    elif model_id == "lmms-lab/llava-next-110b":
        text_model_id = "Qwen/Qwen1.5-110B-Chat"
        image_token_index = 151646
    else:
        text_model_id = "models/Meta-Llama-3-8B"
        image_token_index = 128256

    vision_model_id = data["mm_vision_tower"]

    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    use_fast = False if model_id == "liuhaotian/llava-v1.6-34b" else True
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=use_fast)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    if model_id in ("liuhaotian/llava-v1.6-mistral-7b", "lmms-lab/llama3-llava-next-8b"):
        # Mistral-7B doesn't have a padding token set yet
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = LlavaNextImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)

    config = LlavaNextConfig(
        text_config=text_config.to_dict(),
        image_grid_pinpoints=image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
        image_token_index=image_token_index,
    )

    with init_empty_weights():
        model = LlavaNextForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict(model_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    # Pad to 64 for performance reasons
    # Qwen-based models have extra unused space in the vocab size already, so no need to resize
    if model_id not in ["lmms-lab/llava-next-72b", "lmms-lab/llava-next-110b"]:
        pad_shape = 64
        vocab_size = config.text_config.vocab_size
        if model_id == "liuhaotian/llava-v1.6-34b":
            # this one has 3 additional tokens, namely <|startoftext|>, <|endoftext|> and <image>
            num_tokens = vocab_size + 3
        else:
            # this one has 2 additional tokens, namely <image> and <pad>
            num_tokens = vocab_size + 2
        model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
        model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
            tuple(
                (
                    dist.sample()
                    for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0])
                )
            ),
            dim=0,
        )
        model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
            tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
            dim=0,
        )

    print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model directory."
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, None)