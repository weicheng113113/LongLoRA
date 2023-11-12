# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
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

import os
import math
import torch
import argparse
import random
import numpy as np
import transformers
from peft import PeftModel

from custom_eval_trainer import CustomEvalPrediction, CustomEvalTrainer
from llama_attn_replace import replace_llama_attn
from torch.distributed import barrier
from pg19_dataset import Pg19Dataset
from transformers import TrainingArguments, BitsAndBytesConfig
import evaluate


class EvalMetric:
    def compute_metrics(self, p: CustomEvalPrediction) -> dict[str, float]:
        predictions = p.predictions.argmax(axis=p.predictions.ndim - 1)
        label_ids = p.label_ids

        accuracy = evaluate.load("accuracy", module_type="metric")
        accuracy_result = accuracy.compute(predictions=predictions.flatten(), references=label_ids.flatten())

        losses = p.extras["losses"]
        perplexity = np.exp(losses.mean())
        return {
            "accuracy": accuracy_result["accuracy"],
            "perplexity": perplexity
        }



def parse_config(cmd_args: list[str] = None):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    # parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    args = parser.parse_args(cmd_args)
    return args


def main(args):
    # torch_dtype = torch.float
    # if torch.cuda.is_available():
    #     torch_dtype = torch.float16

    print("data path", args.data_path)
    print("base model", args.base_model)
    print("peft model", args.peft_model)

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = Pg19Dataset(args.data_path, seq_length=args.seq_len, sliding_window=256)

    if rank == 0:
        barrier()

    print(f"Num validation tokens: {dataset.num_tokens()}, Num validation examples: {len(dataset)}")

    if args.flash_attn:
        replace_llama_attn(use_flash_attn=True, use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.resize_token_embeddings(32001)

    if args.peft_model:
        trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        else:
            raise ValueError("Trainable input embedding and normalization are required.")
        model = PeftModel.from_pretrained(
            model,
            args.peft_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            offload_folder=args.cache_dir,
        )

    eval_metric = EvalMetric()
    trainer = CustomEvalTrainer(
        model=model,
        compute_metrics=eval_metric.compute_metrics,
        args=TrainingArguments(per_device_eval_batch_size=args.batch_size, output_dir="./output/tmp"),)
    result = trainer.evaluate(eval_dataset=dataset)
    print(result)


if __name__ == "__main__":
    args = parse_config()
    main(args)
