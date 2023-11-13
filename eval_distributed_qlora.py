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
from dataclasses import dataclass, field
from typing import Optional

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
        predictions = p.predictions.argmax(axis=p.predictions.ndim - 1)[..., :-1]
        label_ids = p.label_ids[..., 1:]

        accuracy = evaluate.load("accuracy", module_type="metric")
        accuracy_result = accuracy.compute(predictions=predictions.flatten(), references=label_ids.flatten())

        losses = p.extras["losses"]
        perplexity = np.exp(losses.mean())
        return {
            "accuracy": accuracy_result["accuracy"],
            "perplexity": perplexity
        }


@dataclass
class EvalArguments(transformers.TrainingArguments):
    base_model: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    seq_len: int = field(
        default=2048,
        metadata={"help": "context length during evaluation."},
    )
    context_size: int = field(
        default=-1,
        metadata={"help": "context size during fine-tuning."},
    )
    peft_model: Optional[str] = field(default=None)
    flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether use flash attention."},
    )
    data_path: str = field(
        default="./test.bin",
        metadata={"help": "test data path"},
    )
    cache_dir: Optional[str] = field(default="./.cache")


def main(args: list[str] = None):
    parser = transformers.HfArgumentParser((EvalArguments, ))
    eval_args: EvalArguments = parser.parse_args_into_dataclasses(args)[0]

    torch_dtype = torch.float
    if torch.cuda.is_available():
        torch_dtype = torch.float16

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = Pg19Dataset(eval_args.data_path, seq_length=eval_args.seq_len, sliding_window=256)

    if rank == 0:
        barrier()

    if rank == 0 or rank == -1:
        print("data path", eval_args.data_path)
        print("base model", eval_args.base_model)
        print("peft model", eval_args.peft_model)
        print(f"Num validation tokens: {dataset.num_tokens()}, Num validation examples: {len(dataset)}")

    if eval_args.flash_attn:
        replace_llama_attn(use_flash_attn=True, use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        eval_args.base_model,
        cache_dir=eval_args.cache_dir,
        use_cache=False
    )

    context_size = eval_args.context_size if eval_args.context_size > 0 else eval_args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        eval_args.base_model,
        config=config,
        cache_dir=eval_args.cache_dir,
        torch_dtype=torch_dtype,
        # torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    )
    model.resize_token_embeddings(32001)

    if eval_args.peft_model:
        trainable_params = os.path.join(eval_args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        else:
            raise ValueError("Trainable input embedding and normalization are required.")
        model = PeftModel.from_pretrained(
            model,
            eval_args.peft_model,
            torch_dtype=torch_dtype,
            # torch_dtype=torch.bfloat16,
            offload_folder=eval_args.cache_dir,
        )

    eval_metric = EvalMetric()
    trainer = CustomEvalTrainer(
        model=model,
        compute_metrics=eval_metric.compute_metrics,
        args=eval_args)
    result = trainer.evaluate(eval_dataset=dataset)
    print(result)


if __name__ == "__main__":
    args = []
    main(args)
