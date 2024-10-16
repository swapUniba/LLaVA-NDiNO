import os
import json
import torch
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--answer_field", type=str)

    return parser.parse_args()


def main(model_id, data_path, images_dir, answer_field):

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, device_map="cuda:0", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)

    with open(data_path, 'r', encoding='utf8') as f:

        ppls = []

        for l in tqdm(f):

            line_data = json.loads(l)

            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{line_data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            completion = f"{line_data[answer_field]}<|eot_id|>"

            image = Image.open(os.path.join(images_dir, line_data['file_name']))

            inputs = processor(text=prompt + completion, images=image, return_tensors="pt")

            response_token_ids = processor.tokenizer(completion)
            labels_mask = inputs['input_ids'].clone()
            ignore_index = -100

            labels_mask[0, :len(labels_mask[0]) - len(response_token_ids[0]) + 1] = ignore_index
            inputs['labels'] = labels_mask

            with torch.no_grad():
                generate_ids = model(**inputs.to('cuda:0'), return_dict=True)

                ppl = torch.exp(generate_ids.loss).cpu()
                if ppl > 1000:
                    continue
                
                ppls.append(ppl)
        
        print(f"PERPLEXITY FOR MODEL {model_id}: {np.mean(ppls)}")


if __name__ == "__main__":

    args = get_args()
    main(**vars(args))
