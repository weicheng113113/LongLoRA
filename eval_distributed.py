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
import random
import numpy as np
import transformers
from peft import PeftModel

from distributed_evaluator import DistributedEvaluator, EvalMetric
from llama_attn_replace import replace_llama_attn
from pg19_dataset import Pg19Dataset
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics import Accuracy
from torchmetrics.text import Perplexity
from torch.nn import CrossEntropyLoss


class EvalMetricImpl(EvalMetric):
    def __init__(self, vocab_size: int, gpu_id: int):
        self.accuracy = Accuracy(task="multiclass", num_classes=vocab_size).to(gpu_id)
        self.perplexity = Perplexity(ignore_index=CrossEntropyLoss().ignore_index).to(gpu_id)
        self.last_loss = 0.0

    def add(self, logits: torch.FloatTensor, labels: torch.LongTensor, model_output: object) -> dict[str, object]:
        shift_predictions = logits.argmax(dim=-1)[..., :-1]
        shift_labels = labels[..., 1:]

        current_accuracy = self.accuracy.forward(preds=shift_predictions, target=shift_labels)

        shift_logits = logits[..., :-1, :]
        current_perplexity = self.perplexity.forward(preds=shift_logits, target=shift_labels)

        self.last_loss = model_output["loss"].item()
        return {
            "accuracy": current_accuracy.item(),
            "perplexity": current_perplexity.item(),
            "loss": self.last_loss
        }

    def compute(self) -> dict[str, object]:
        current_accuracy = self.accuracy.compute()
        current_perplexity = self.perplexity.compute()
        return {
            "accuracy": current_accuracy.item(),
            "perplexity": current_perplexity.item(),
            "loss": self.last_loss
        }


@dataclass
class EvalArguments:
    batch_size: int = field(
        default=1,
        metadata={"help": "batch size."},
    )
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
    progress_bar_fresh_rate: int = field(
        default=10,
        metadata={"help": "progress bar metrics fresh rate."},
    )


def run_eval(args: EvalArguments):
    torch_dtype = torch.float16

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = Pg19Dataset(args.data_path, seq_length=args.seq_len, sliding_window=256)
    if args.flash_attn:
        replace_llama_attn(use_flash_attn=True, use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        use_cache=False
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype)
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
            torch_dtype=torch_dtype,
            offload_folder=args.cache_dir,
        )

    [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in ["lm_head"]])]

    gpu_id = int(os.environ["LOCAL_RANK"])
    model.to(gpu_id)

    evaluator = DistributedEvaluator(
        model=model,
        batch_size=args.batch_size,
        refresh_rate=args.progress_bar_fresh_rate,
        gpu_id=gpu_id)

    if evaluator.is_first_device():
        print("data path", args.data_path)
        print("base model", args.base_model)
        print("peft model", args.peft_model)
        print(f"Num validation tokens: {dataset.num_tokens()}, Num validation examples: {len(dataset)}")

    eval_metric = EvalMetricImpl(vocab_size=config.vocab_size, gpu_id=gpu_id)
    result = evaluator.evaluate(dataset, eval_metric)
    if evaluator.is_first_device():
        print(result)


def ddp_setup():
    init_process_group(backend="nccl")


# def ddp_setup2(rank: int, world_size: int):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)


# def main(rank: int, world_size: int, cmd_args: list[str] = None):
#     ddp_setup2(rank, world_size)
#     # ddp_setup()
#     parser = transformers.HfArgumentParser((EvalArguments, ))
#     args: EvalArguments = parser.parse_args_into_dataclasses(cmd_args)[0]
#     try:
#         run_eval(rank, args)
#     finally:
#         destroy_process_group()


def main(cmd_args: list[str] = None):
    ddp_setup()
    parser = transformers.HfArgumentParser((EvalArguments, ))
    args: EvalArguments = parser.parse_args_into_dataclasses(cmd_args)[0]
    try:
        run_eval(args)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
