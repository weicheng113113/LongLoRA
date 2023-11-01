import os

from datasets import load_dataset
from fine_tune import tokenize_fn, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN
import transformers
from functools import partial


def prepare(model_name_or_path: str, cache_dir: str, model_max_length: int, num_proc: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=cache_dir)
    dataset = dataset.map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=["text", "meta"])

    print(dataset)


def main():
    prepare(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        cache_dir="./data/.cache",
        model_max_length=8192,
        num_proc=4)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    main()
